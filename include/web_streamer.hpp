#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <cstring>
#include <opencv2/opencv.hpp>

// POSIX Networking Headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <poll.h>

struct WebStats {
    int frameId = 0;
    double capFps = 0.0;
    double infFps = 0.0;
    double capWorkMs = 0.0;
    double trtMs = 0.0;
    uint64_t packetDrops = 0;
    int numCams = 0;
};

struct CameraDetail {
    int index = 0;
    std::string serial;
    std::string model;
    std::string ip;
};

class MJPEGWebServer {
private:
    int m_port;
    int m_serverFd;
    std::atomic<bool> m_running;
    std::thread m_serverThread;

    std::mutex m_frameMutex;
    std::vector<cv::Mat> m_latestPostImages;
    std::vector<cv::Mat> m_latestPreImages;
    WebStats m_latestStats;

    std::mutex m_cameraMutex;
    std::vector<CameraDetail> m_cameraDetails;

    static bool sendAll(int clientFd, const char* data, size_t length) {
        size_t totalSent = 0;
        while (totalSent < length) {
            ssize_t sent = send(clientFd, data + totalSent, length - totalSent, MSG_NOSIGNAL);
            if (sent <= 0) {
                return false;
            }
            totalSent += sent;
        }
        return true;
    }

    static bool sendString(int clientFd, const std::string& str) {
        return sendAll(clientFd, str.c_str(), str.length());
    }

    // Helper to ensure an image is 3-channel BGR for display/grid composition
    static cv::Mat ensureBGR(const cv::Mat& src) {
        if (src.empty()) return cv::Mat();
        if (src.channels() == 3) {
            return src;
        }
        cv::Mat bgr;
        if (src.channels() == 1) {
            try {
                // If BayerRG8 (OpenCV calls BayerRG8 pattern "BG")
                cv::cvtColor(src, bgr, cv::COLOR_BayerBG2BGR);
            } catch (...) {
                cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
            }
            return bgr;
        }
        return src;
    }

    // Dynamic grid layout builder for 1 to 8 cameras
    static cv::Mat composeGrid(const std::vector<cv::Mat>& images, int frameId, uint64_t drops, const std::string& prefix = "CAM ") {
        if (images.empty()) return cv::Mat();

        int n = static_cast<int>(images.size());
        cv::Scalar okCol(0, 255, 0);
        cv::Scalar dropCol(0, 0, 255);
        cv::Scalar tagCol = (drops > 0) ? dropCol : okCol;

        // 1 Camera
        if (n == 1) {
            cv::Mat bgr = ensureBGR(images[0]);
            cv::Mat out;
            cv::resize(bgr, out, cv::Size(960, 600));
            std::string label = prefix + "0 | #" + std::to_string(frameId) + " | Drops: " + std::to_string(drops);
            cv::putText(out, label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, tagCol, 2);
            return out;
        }

        // 2 Cameras: 1 Row x 2 Columns (1280 x 400)
        if (n == 2) {
            cv::Mat bgr0 = ensureBGR(images[0]);
            cv::Mat bgr1 = ensureBGR(images[1]);
            cv::Mat r0, r1, out;
            cv::resize(bgr0, r0, cv::Size(640, 400));
            cv::resize(bgr1, r1, cv::Size(640, 400));

            cv::putText(r0, prefix + "0 | #" + std::to_string(frameId), cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, tagCol, 2);
            cv::putText(r1, prefix + "1 | #" + std::to_string(frameId), cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, tagCol, 2);

            cv::hconcat(r0, r1, out);
            return out;
        }

        // 3 or 4 Cameras: 2 Rows x 2 Columns (1280 x 800)
        // 5 to 8 Cameras: 2 Rows x 4 Columns (1600 x 500)
        int cols = (n <= 4) ? 2 : 4;
        int rows = 2;
        int tileW = (n <= 4) ? 640 : 400;
        int tileH = (n <= 4) ? 400 : 250;

        std::vector<cv::Mat> tiles(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            if (i < n && !images[i].empty()) {
                cv::Mat bgr = ensureBGR(images[i]);
                cv::resize(bgr, tiles[i], cv::Size(tileW, tileH));
                std::string label = prefix + std::to_string(i) + " | #" + std::to_string(frameId);
                cv::putText(tiles[i], label, cv::Point(15, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, tagCol, 2);
            } else {
                tiles[i] = cv::Mat::zeros(cv::Size(tileW, tileH), CV_8UC3);
                std::string emptyLabel = (i < 8) ? prefix + std::to_string(i) + " (OFF)" : "EMPTY";
                cv::putText(tiles[i], emptyLabel, cv::Point(tileW / 4, tileH / 2),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(100, 100, 100), 2);
            }
        }

        std::vector<cv::Mat> rowMats(rows);
        for (int r = 0; r < rows; r++) {
            std::vector<cv::Mat> rowTiles;
            for (int c = 0; c < cols; c++) {
                rowTiles.push_back(tiles[r * cols + c]);
            }
            cv::hconcat(rowTiles, rowMats[r]);
        }

        cv::Mat gridOut;
        cv::vconcat(rowMats, gridOut);
        return gridOut;
    }

    static cv::Mat composeCombined(const std::vector<cv::Mat>& preImages,
                                   const std::vector<cv::Mat>& postImages,
                                   int frameId, uint64_t drops) {
        cv::Mat preGrid = composeGrid(preImages, frameId, drops, "PRE CAM ");
        cv::Mat postGrid = composeGrid(postImages, frameId, drops, "DET CAM ");
        if (preGrid.empty()) return postGrid;
        if (postGrid.empty()) return preGrid;

        if (preGrid.cols != postGrid.cols) {
            cv::resize(preGrid, preGrid, cv::Size(postGrid.cols, preGrid.rows));
        }

        cv::Mat out;
        cv::vconcat(preGrid, postGrid, out);
        return out;
    }

    void serveIndexHtml(int clientFd) {
        std::string html = 
R"rawhtml(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FourEyes | Real-time Multi-Camera Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; padding: 24px; }
        .container { max-width: 1600px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #1e293b; padding-bottom: 16px; }
        h1 { font-size: 22px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 12px; }
        .pill { background: #10b981; color: #042f2e; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 9999px; letter-spacing: 0.5px; }
        .grid-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .stat-card { background: #131d31; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 18px; }
        .stat-label { font-size: 11px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.6px; margin-bottom: 6px; }
        .stat-val { font-size: 24px; font-weight: 700; color: #f8fafc; }
        .val-ok { color: #34d399; }
        .val-warn { color: #f87171; }
        .view-controls { display: flex; gap: 10px; margin-bottom: 16px; }
        .btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8; padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s ease; }
        .btn:hover { background: #334155; color: #f8fafc; }
        .btn.active { background: #0284c7; color: #ffffff; border-color: #38bdf8; }
        .streams-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(640px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stream-card { background: #020617; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); }
        .stream-header { padding: 10px 18px; background: #0f172a; font-size: 13px; font-weight: 600; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; }
        .badge { font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.5px; }
        .badge-purple { background: #4c1d95; color: #c4b5fd; }
        .badge-green { background: #064e3b; color: #6ee7b7; }
        .stream-img { width: 100%; height: auto; display: block; background: #000; min-height: 380px; object-fit: contain; }
        .nav-links { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: #64748b; align-items: center; }
        a { color: #38bdf8; text-decoration: none; font-weight: 500; }
        a:hover { text-decoration: underline; }

        /* Camera Details & Copyable Serials Panel */
        .cam-panel { background: #131d31; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px; }
        .cam-panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px; }
        .cam-panel-title { font-size: 12px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.6px; display: flex; align-items: center; gap: 8px; }
        .cam-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
        .cam-item { background: #0b1120; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
        .cam-item-top { display: flex; justify-content: space-between; align-items: center; }
        .cam-id-badge { background: #0284c7; color: #fff; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; }
        .cam-model { font-size: 12px; color: #94a3b8; font-weight: 500; }
        .cam-serial-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
        .cam-serial-code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 14px; font-weight: 700; color: #34d399; background: #020617; padding: 6px 10px; border-radius: 6px; border: 1px solid #1e293b; letter-spacing: 0.8px; flex-grow: 1; user-select: all; cursor: pointer; }
        .cam-serial-code:hover { border-color: #38bdf8; background: #091224; }
        .btn-copy { background: #1e293b; border: 1px solid #334155; color: #cbd5e1; padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s ease; white-space: nowrap; }
        .btn-copy:hover { background: #38bdf8; color: #0f172a; border-color: #38bdf8; }
        .btn-copy.copied { background: #10b981; color: #042f2e; border-color: #34d399; }
        .btn-action { background: #1e293b; border: 1px solid #334155; color: #cbd5e1; padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s ease; }
        .btn-action:hover { background: #0284c7; color: #fff; border-color: #38bdf8; }
        .btn-action.copied { background: #10b981; color: #042f2e; border-color: #34d399; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>
                Lucid Vision Labs Perception Stream
                <span class="pill" id="live-pill">CONNECTED</span>
            </h1>
            <span style="font-size: 13px; color: #64748b;">Jetson AGX Orin | IEEE 1588 PTP Sync</span>
        </header>

        <div class="grid-stats">
            <div class="stat-card">
                <div class="stat-label">Frame Counter</div>
                <div class="stat-val" id="st-frame">--</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Capture Rate</div>
                <div class="stat-val" id="st-cap-fps">-- FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Inference Rate</div>
                <div class="stat-val" id="st-inf-fps">-- FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">TRT Latency</div>
                <div class="stat-val" id="st-trt">-- ms</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Packet Drops</div>
                <div class="stat-val val-ok" id="st-drops">0</div>
            </div>
        </div>

        <!-- Camera Hardware & Serial Numbers Panel -->
        <div class="cam-panel" id="cam-panel">
            <div class="cam-panel-header">
                <div class="cam-panel-title">
                    <span>Connected Cameras &amp; Serial Numbers</span>
                    <span style="font-size: 11px; color: #64748b; font-weight: 400; text-transform: none;">(Click any serial number or button to copy)</span>
                </div>
                <div style="display: flex; gap: 8px;">
                    <button class="btn-action" id="btn-copy-all" onclick="copyAllSerials()">Copy All Serials</button>
                    <button class="btn-action" id="btn-copy-json" onclick="copySerialsJson()">Copy JSON Map</button>
                </div>
            </div>
            <div class="cam-grid" id="cam-grid-container">
                <div style="color: #64748b; font-size: 13px; padding: 8px;">Discovering camera hardware serial numbers...</div>
            </div>
        </div>

        <div class="view-controls">
            <button class="btn active" id="btn-dual" onclick="setView('dual')">Dual View (Pre + Post)</button>
            <button class="btn" id="btn-post" onclick="setView('post')">Detection Results Only</button>
            <button class="btn" id="btn-pre" onclick="setView('pre')">Pre-Inference (Camera) Only</button>
        </div>

        <div class="streams-container" id="streams-grid">
            <div class="stream-card" id="card-pre">
                <div class="stream-header">
                    <span id="title-pre">Pre-Inference Camera Feed</span>
                    <span class="badge badge-purple">PRE-INFERENCE (CAMERA)</span>
                </div>
                <img class="stream-img" src="/pre_stream" alt="Pre-Inference Camera Stream" />
            </div>

            <div class="stream-card" id="card-post">
                <div class="stream-header">
                    <span id="title-post">Post-Inference Detection Results</span>
                    <span class="badge badge-green">YOLOv10 DETECTIONS</span>
                </div>
                <img class="stream-img" src="/stream" alt="Detection Results Stream" />
            </div>
        </div>

        <div class="nav-links">
            <span>Direct Feeds:</span>
            <span><a href="/stream" target="_blank">Detections Grid</a></span>
            <span>&bull;</span>
            <span><a href="/pre_stream" target="_blank">Pre-Inference Grid</a></span>
            <span>&bull;</span>
            <span><a href="/combined_stream" target="_blank">Combined Dual Grid</a></span>
            <span>&bull;</span>
            <span id="direct-links"></span>
            <span>&bull;</span>
            <span><a href="/snapshot" target="_blank">Snapshot (Detections)</a></span>
            <span>&bull;</span>
            <span><a href="/pre_snapshot" target="_blank">Snapshot (Pre-Inf)</a></span>
            <span>&bull;</span>
            <span><a href="/stats" target="_blank">Raw Stats JSON</a></span>
        </div>
    </div>

    <script>
        let lastNumCams = -1;
        let currentCameras = [];

        function copyText(text, btn) {
            function showSuccess() {
                if (!btn) return;
                const orig = btn.innerText;
                btn.innerText = 'Copied!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.innerText = orig;
                    btn.classList.remove('copied');
                }, 1500);
            }
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(showSuccess).catch(() => fallbackCopy(text, showSuccess));
            } else {
                fallbackCopy(text, showSuccess);
            }
        }

        function fallbackCopy(text, cb) {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            try { document.execCommand('copy'); if (cb) cb(); } catch (e) {}
            document.body.removeChild(ta);
        }

        function copyAllSerials() {
            if (!currentCameras.length) return;
            const txt = currentCameras.map(c => `CAM ${c.index}: ${c.serial} (${c.model || 'TRI023S-C'})`).join('\n');
            copyText(txt, document.getElementById('btn-copy-all'));
        }

        function copySerialsJson() {
            if (!currentCameras.length) return;
            const obj = {};
            currentCameras.forEach(c => { obj[`CAM_${c.index}`] = c.serial; });
            copyText(JSON.stringify(obj, null, 2), document.getElementById('btn-copy-json'));
        }

        function setView(view) {
            const cardPre = document.getElementById('card-pre');
            const cardPost = document.getElementById('card-post');
            const grid = document.getElementById('streams-grid');
            const btnDual = document.getElementById('btn-dual');
            const btnPost = document.getElementById('btn-post');
            const btnPre = document.getElementById('btn-pre');

            btnDual.className = 'btn' + (view === 'dual' ? ' active' : '');
            btnPost.className = 'btn' + (view === 'post' ? ' active' : '');
            btnPre.className = 'btn' + (view === 'pre' ? ' active' : '');

            if (view === 'dual') {
                cardPre.style.display = 'block';
                cardPost.style.display = 'block';
                grid.style.gridTemplateColumns = 'repeat(auto-fit, minmax(640px, 1fr))';
            } else if (view === 'post') {
                cardPre.style.display = 'none';
                cardPost.style.display = 'block';
                grid.style.gridTemplateColumns = '1fr';
            } else if (view === 'pre') {
                cardPre.style.display = 'block';
                cardPost.style.display = 'none';
                grid.style.gridTemplateColumns = '1fr';
            }
        }

        function refreshStats() {
            fetch('/stats')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('st-frame').innerText = '#' + data.frameId;
                    document.getElementById('st-cap-fps').innerText = data.capFps.toFixed(1) + ' FPS';
                    document.getElementById('st-inf-fps').innerText = data.infFps.toFixed(1) + ' FPS';
                    document.getElementById('st-trt').innerText = data.trtMs.toFixed(1) + ' ms';
                    
                    const dropEl = document.getElementById('st-drops');
                    dropEl.innerText = data.packetDrops;
                    if (data.packetDrops > 0) {
                        dropEl.className = 'stat-val val-warn';
                    } else {
                        dropEl.className = 'stat-val val-ok';
                    }

                    if (data.numCams !== lastNumCams) {
                        lastNumCams = data.numCams;
                        document.getElementById('live-pill').innerText = data.numCams + ' CAMERAS ACTIVE';
                        document.getElementById('title-pre').innerText = 'Pre-Inference Input (' + data.numCams + ' Cams)';
                        document.getElementById('title-post').innerText = 'Post-Inference YOLOv10 (' + data.numCams + ' Cams)';

                        let linkHtml = 'Cameras: ';
                        for (let i = 0; i < data.numCams; i++) {
                            linkHtml += ` <a href="/cam${i}" target="_blank">Cam ${i} (Det)</a> &bull; <a href="/cam${i}_pre" target="_blank">Cam ${i} (Pre)</a>`;
                            if (i < data.numCams - 1) linkHtml += ' | ';
                        }
                        document.getElementById('direct-links').innerHTML = linkHtml;
                    }

                    if (data.cameras && JSON.stringify(data.cameras) !== JSON.stringify(currentCameras)) {
                        currentCameras = data.cameras;
                        let html = '';
                        for (const c of currentCameras) {
                            html += `
                            <div class="cam-item">
                                <div class="cam-item-top">
                                    <span class="cam-id-badge">CAM ${c.index}</span>
                                    <span class="cam-model">${c.model || 'Lucid TRI023S-C'}</span>
                                </div>
                                <div class="cam-serial-row">
                                    <code class="cam-serial-code" title="Click to copy" onclick="copyText('${c.serial}', this)">${c.serial}</code>
                                    <button class="btn-copy" onclick="copyText('${c.serial}', this)">Copy</button>
                                </div>
                                <div style="font-size: 11px; color: #64748b;">IP: ${c.ip || 'N/A'}</div>
                            </div>`;
                        }
                        if (currentCameras.length === 0) {
                            html = '<div style="color: #64748b; font-size: 13px; padding: 8px;">No camera hardware metadata registered.</div>';
                        }
                        document.getElementById('cam-grid-container').innerHTML = html;
                    }
                })
                .catch(() => {});
        }
        setInterval(refreshStats, 1000);
        refreshStats();
    </script>
</body>
</html>)rawhtml";

        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n"
                 << "Content-Type: text/html\r\n"
                 << "Content-Length: " << html.length() << "\r\n"
                 << "Connection: close\r\n\r\n"
                 << html;
        sendString(clientFd, response.str());
        close(clientFd);
    }

    void serveStatsJson(int clientFd) {
        WebStats s;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            s = m_latestStats;
        }

        std::vector<CameraDetail> cams;
        {
            std::lock_guard<std::mutex> lock(m_cameraMutex);
            cams = m_cameraDetails;
        }

        std::ostringstream json;
        json << "{"
             << "\"frameId\":" << s.frameId << ","
             << "\"capFps\":" << s.capFps << ","
             << "\"infFps\":" << s.infFps << ","
             << "\"capWorkMs\":" << s.capWorkMs << ","
             << "\"trtMs\":" << s.trtMs << ","
             << "\"packetDrops\":" << s.packetDrops << ","
             << "\"numCams\":" << s.numCams << ","
             << "\"cameras\":[";

        for (size_t i = 0; i < cams.size(); ++i) {
            json << "{"
                 << "\"index\":" << cams[i].index << ","
                 << "\"serial\":\"" << cams[i].serial << "\","
                 << "\"model\":\"" << cams[i].model << "\","
                 << "\"ip\":\"" << cams[i].ip << "\""
                 << "}";
            if (i + 1 < cams.size()) json << ",";
        }
        json << "]}";

        std::string jsonStr = json.str();
        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n"
                 << "Content-Type: application/json\r\n"
                 << "Access-Control-Allow-Origin: *\r\n"
                 << "Content-Length: " << jsonStr.length() << "\r\n"
                 << "Connection: close\r\n\r\n"
                 << jsonStr;
        sendString(clientFd, response.str());
        close(clientFd);
    }

    void serveSnapshot(int clientFd, int streamType = 0) {
        cv::Mat snap;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            if (streamType == 1 && !m_latestPreImages.empty()) {
                snap = composeGrid(m_latestPreImages, m_latestStats.frameId, m_latestStats.packetDrops, "PRE CAM ");
            } else if (streamType == 2 && (!m_latestPostImages.empty() || !m_latestPreImages.empty())) {
                snap = composeCombined(m_latestPreImages, m_latestPostImages, m_latestStats.frameId, m_latestStats.packetDrops);
            } else if (!m_latestPostImages.empty()) {
                snap = composeGrid(m_latestPostImages, m_latestStats.frameId, m_latestStats.packetDrops, "DET CAM ");
            }
        }

        if (snap.empty()) {
            std::string emptyResp = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            sendString(clientFd, emptyResp);
            close(clientFd);
            return;
        }

        std::vector<uchar> jpegBuf;
        cv::imencode(".jpg", snap, jpegBuf, {cv::IMWRITE_JPEG_QUALITY, 85});

        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n"
                 << "Content-Type: image/jpeg\r\n"
                 << "Content-Length: " << jpegBuf.size() << "\r\n"
                 << "Connection: close\r\n\r\n";
        sendString(clientFd, response.str());
        sendAll(clientFd, reinterpret_cast<const char*>(jpegBuf.data()), jpegBuf.size());
        close(clientFd);
    }

    void serveMjpegStream(int clientFd, int camIdx, int streamType = 0) {
        std::string streamHeader = 
            "HTTP/1.1 200 OK\r\n"
            "Server: FourEyes-MJPEG\r\n"
            "Cache-Control: no-cache, no-store, must-revalidate\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

        if (!sendString(clientFd, streamHeader)) {
            close(clientFd);
            return;
        }

        std::vector<uchar> jpegBuf;
        std::vector<int> encodeParams = {cv::IMWRITE_JPEG_QUALITY, 70};
        int lastServedFrameId = -1;

        while (m_running) {
            cv::Mat displayFrame;

            {
                std::lock_guard<std::mutex> lock(m_frameMutex);
                const auto& targetImages = (streamType == 1) ? m_latestPreImages : m_latestPostImages;
                if (!targetImages.empty() && m_latestStats.frameId != lastServedFrameId) {
                    int frameId = m_latestStats.frameId;
                    uint64_t drops = m_latestStats.packetDrops;

                    if (streamType == 2) {
                        displayFrame = composeCombined(m_latestPreImages, m_latestPostImages, frameId, drops);
                    } else if (camIdx >= 0 && camIdx < (int)targetImages.size()) {
                        cv::Mat bgr = ensureBGR(targetImages[camIdx]);
                        cv::resize(bgr, displayFrame, cv::Size(960, 600));
                        std::string prefix = (streamType == 1) ? "PRE CAM " : "DET CAM ";
                        std::string label = prefix + std::to_string(camIdx) + " | #" + std::to_string(frameId) + 
                                            " | Drops: " + std::to_string(drops);
                        cv::putText(displayFrame, label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                                    (drops > 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0)), 2);
                    } else {
                        std::string prefix = (streamType == 1) ? "PRE CAM " : "DET CAM ";
                        displayFrame = composeGrid(targetImages, frameId, drops, prefix);
                    }
                    lastServedFrameId = frameId;
                }
            }

            if (!displayFrame.empty()) {
                jpegBuf.clear();
                cv::imencode(".jpg", displayFrame, jpegBuf, encodeParams);

                std::ostringstream partHeader;
                partHeader << "--frame\r\n"
                           << "Content-Type: image/jpeg\r\n"
                           << "Content-Length: " << jpegBuf.size() << "\r\n\r\n";

                std::string headerStr = partHeader.str();
                if (!sendAll(clientFd, headerStr.c_str(), headerStr.length())) break;
                if (!sendAll(clientFd, reinterpret_cast<const char*>(jpegBuf.data()), jpegBuf.size())) break;
                if (!sendString(clientFd, "\r\n")) break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(25)); // ~40 FPS stream pacing
        }

        close(clientFd);
    }

    void handleClient(int clientFd) {
        struct timeval tv;
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        setsockopt(clientFd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));

        char buffer[2048] = {0};
        ssize_t bytesRead = read(clientFd, buffer, sizeof(buffer) - 1);
        if (bytesRead <= 0) {
            close(clientFd);
            return;
        }

        std::string request(buffer);
        std::istringstream reqStream(request);
        std::string method, path;
        reqStream >> method >> path;

        if (path == "/" || path == "/index.html") {
            serveIndexHtml(clientFd);
        } else if (path == "/stats") {
            serveStatsJson(clientFd);
        } else if (path == "/snapshot") {
            serveSnapshot(clientFd, 0);
        } else if (path == "/pre_snapshot") {
            serveSnapshot(clientFd, 1);
        } else if (path == "/combined_snapshot") {
            serveSnapshot(clientFd, 2);
        } else if (path == "/stream" || path == "/post_stream") {
            serveMjpegStream(clientFd, -1, 0);
        } else if (path == "/pre_stream") {
            serveMjpegStream(clientFd, -1, 1);
        } else if (path == "/combined_stream" || path == "/dual_stream") {
            serveMjpegStream(clientFd, -1, 2);
        } else if (path.rfind("/cam", 0) == 0) {
            bool isPre = (path.find("_pre") != std::string::npos);
            int camIdx = 0;
            try {
                size_t start = 4;
                size_t end = path.find('_');
                if (end == std::string::npos) end = path.length();
                camIdx = std::stoi(path.substr(start, end - start));
            } catch (...) {
                camIdx = 0;
            }
            serveMjpegStream(clientFd, camIdx, isPre ? 1 : 0);
        } else {
            std::string notFound = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            sendString(clientFd, notFound);
            close(clientFd);
        }
    }

    void listenerLoop() {
        signal(SIGPIPE, SIG_IGN); // Ignore SIGPIPE to prevent app crash if browser disconnects

        while (m_running) {
            struct pollfd pfd;
            pfd.fd = m_serverFd;
            pfd.events = POLLIN;

            int ret = poll(&pfd, 1, 500); // 500 ms timeout to check m_running
            if (ret <= 0) {
                continue;
            }

            struct sockaddr_in clientAddr;
            socklen_t clientLen = sizeof(clientAddr);
            int clientFd = accept(m_serverFd, (struct sockaddr*)&clientAddr, &clientLen);

            if (clientFd >= 0) {
                // Handle each client in a lightweight detached thread
                std::thread([this, clientFd]() {
                    this->handleClient(clientFd);
                }).detach();
            }
        }
    }

public:
    explicit MJPEGWebServer(int port = 8080)
        : m_port(port), m_serverFd(-1), m_running(false) {}

    ~MJPEGWebServer() {
        stop();
    }

    bool start() {
        if (m_running) return true;

        m_serverFd = socket(AF_INET, SOCK_STREAM, 0);
        if (m_serverFd < 0) {
            std::cerr << "[Web Server] Failed to create socket\n";
            return false;
        }

        int opt = 1;
        setsockopt(m_serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in serverAddr;
        std::memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(m_port);

        if (bind(m_serverFd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "[Web Server] Failed to bind to port " << m_port << "\n";
            close(m_serverFd);
            m_serverFd = -1;
            return false;
        }

        if (listen(m_serverFd, 10) < 0) {
            std::cerr << "[Web Server] Failed to listen on socket\n";
            close(m_serverFd);
            m_serverFd = -1;
            return false;
        }

        m_running = true;
        m_serverThread = std::thread(&MJPEGWebServer::listenerLoop, this);
        std::cout << "[Web Server] Live monitor listening on http://0.0.0.0:" << m_port << std::endl;
        return true;
    }

    void stop() {
        if (!m_running) return;
        m_running = false;

        if (m_serverFd >= 0) {
            close(m_serverFd);
            m_serverFd = -1;
        }

        if (m_serverThread.joinable()) {
            m_serverThread.join();
        }
        std::cout << "[Web Server] Stopped cleanly.\n";
    }

    void updateFrames(const std::vector<cv::Mat>& preImages,
                      const std::vector<cv::Mat>& postImages,
                      const WebStats& stats) {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_latestPreImages.clear();
        m_latestPreImages.reserve(preImages.size());
        for (const auto& img : preImages) {
            m_latestPreImages.push_back(img.clone());
        }

        m_latestPostImages.clear();
        m_latestPostImages.reserve(postImages.size());
        for (const auto& img : postImages) {
            m_latestPostImages.push_back(img.clone());
        }

        m_latestStats = stats;
    }

    void updateFrame(const std::vector<cv::Mat>& bgrImages, const WebStats& stats) {
        updateFrames(bgrImages, bgrImages, stats);
    }

    void setCameraDetails(const std::vector<CameraDetail>& details) {
        std::lock_guard<std::mutex> lock(m_cameraMutex);
        m_cameraDetails = details;
    }
};
