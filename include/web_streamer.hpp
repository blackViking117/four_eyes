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

class MJPEGWebServer {
private:
    int m_port;
    int m_serverFd;
    std::atomic<bool> m_running;
    std::thread m_serverThread;

    std::mutex m_frameMutex;
    std::vector<cv::Mat> m_latestImages;
    WebStats m_latestStats;

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

    // Dynamic grid layout builder for 1 to 8 cameras
    static cv::Mat composeGrid(const std::vector<cv::Mat>& images, int frameId, uint64_t drops) {
        if (images.empty()) return cv::Mat();

        int n = static_cast<int>(images.size());
        cv::Scalar okCol(0, 255, 0);
        cv::Scalar dropCol(0, 0, 255);
        cv::Scalar tagCol = (drops > 0) ? dropCol : okCol;

        // 1 Camera
        if (n == 1) {
            cv::Mat out;
            cv::resize(images[0], out, cv::Size(960, 600));
            std::string label = "CAM 0 | #" + std::to_string(frameId) + " | Drops: " + std::to_string(drops);
            cv::putText(out, label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, tagCol, 2);
            return out;
        }

        // 2 Cameras: 1 Row x 2 Columns (1280 x 400)
        if (n == 2) {
            cv::Mat r0, r1, out;
            cv::resize(images[0], r0, cv::Size(640, 400));
            cv::resize(images[1], r1, cv::Size(640, 400));

            cv::putText(r0, "CAM 0 | #" + std::to_string(frameId), cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, tagCol, 2);
            cv::putText(r1, "CAM 1 | #" + std::to_string(frameId), cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, tagCol, 2);

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
                cv::resize(images[i], tiles[i], cv::Size(tileW, tileH));
                std::string label = "CAM " + std::to_string(i) + " | #" + std::to_string(frameId);
                cv::putText(tiles[i], label, cv::Point(15, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, tagCol, 2);
            } else {
                tiles[i] = cv::Mat::zeros(cv::Size(tileW, tileH), CV_8UC3);
                std::string emptyLabel = (i < 8) ? "CAM " + std::to_string(i) + " (OFF)" : "EMPTY";
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
        .container { max-width: 1440px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #1e293b; padding-bottom: 16px; }
        h1 { font-size: 22px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 12px; }
        .pill { background: #10b981; color: #042f2e; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 9999px; letter-spacing: 0.5px; }
        .grid-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .stat-card { background: #131d31; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 18px; }
        .stat-label { font-size: 11px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.6px; margin-bottom: 6px; }
        .stat-val { font-size: 24px; font-weight: 700; color: #f8fafc; }
        .val-ok { color: #34d399; }
        .val-warn { color: #f87171; }
        .stream-card { background: #020617; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); }
        .stream-header { padding: 10px 18px; background: #0f172a; font-size: 13px; font-weight: 600; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; }
        .stream-img { width: 100%; height: auto; display: block; background: #000; min-height: 400px; object-fit: contain; }
        .nav-links { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #64748b; align-items: center; }
        a { color: #38bdf8; text-decoration: none; font-weight: 500; }
        a:hover { text-decoration: underline; }
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

        <div class="stream-card">
            <div class="stream-header">
                <span id="stream-title">Synchronized Multi-Camera Grid</span>
                <span style="font-size: 12px; color: #38bdf8;">BayerRG8 &rarr; BGR &rarr; Dynamic Grid MJPEG</span>
            </div>
            <img class="stream-img" src="/stream" alt="Live Camera Video Stream" />
        </div>

        <div class="nav-links">
            <span id="direct-links">Direct Feeds: <a href="/stream" target="_blank">All Grid</a></span>
            <span>&bull;</span>
            <span><a href="/snapshot" target="_blank">Full Grid Snapshot</a></span>
            <span>&bull;</span>
            <span><a href="/stats" target="_blank">Raw Stats JSON</a></span>
        </div>
    </div>

    <script>
        let lastNumCams = -1;

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
                        document.getElementById('stream-title').innerText = 'Synchronized ' + data.numCams + '-Camera Grid';

                        let linkHtml = 'Direct Feeds: <a href="/stream" target="_blank">All Grid</a>';
                        for (let i = 0; i < data.numCams; i++) {
                            linkHtml += ` &bull; <a href="/cam${i}" target="_blank">Camera ${i}</a>`;
                        }
                        document.getElementById('direct-links').innerHTML = linkHtml;
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

        std::ostringstream json;
        json << "{"
             << "\"frameId\":" << s.frameId << ","
             << "\"capFps\":" << s.capFps << ","
             << "\"infFps\":" << s.infFps << ","
             << "\"capWorkMs\":" << s.capWorkMs << ","
             << "\"trtMs\":" << s.trtMs << ","
             << "\"packetDrops\":" << s.packetDrops << ","
             << "\"numCams\":" << s.numCams
             << "}";

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

    void serveSnapshot(int clientFd) {
        cv::Mat snap;
        {
            std::lock_guard<std::mutex> lock(m_frameMutex);
            if (!m_latestImages.empty()) {
                snap = composeGrid(m_latestImages, m_latestStats.frameId, m_latestStats.packetDrops);
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

    void serveMjpegStream(int clientFd, int camIdx) {
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
                if (!m_latestImages.empty() && m_latestStats.frameId != lastServedFrameId) {
                    int frameId = m_latestStats.frameId;
                    uint64_t drops = m_latestStats.packetDrops;

                    if (camIdx >= 0 && camIdx < (int)m_latestImages.size()) {
                        // Stream a single specific camera
                        cv::resize(m_latestImages[camIdx], displayFrame, cv::Size(960, 600));
                        std::string label = "CAM " + std::to_string(camIdx) + " | #" + std::to_string(frameId) + 
                                            " | Drops: " + std::to_string(drops);
                        cv::putText(displayFrame, label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                                    (drops > 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0)), 2);
                    } else {
                        // Stream multi-camera composite grid
                        displayFrame = composeGrid(m_latestImages, frameId, drops);
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
                if (!sendAll(clientFd, headerStr.data(), headerStr.size())) {
                    break;
                }
                if (!sendAll(clientFd, reinterpret_cast<const char*>(jpegBuf.data()), jpegBuf.size())) {
                    break;
                }
                if (!sendString(clientFd, "\r\n")) {
                    break;
                }
            }

            // Stream throttle for smooth display (~25-30 FPS)
            std::this_thread::sleep_for(std::chrono::milliseconds(35));
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
            serveSnapshot(clientFd);
        } else if (path.rfind("/cam", 0) == 0) {
            // Parses any /cam0, /cam1, /cam2, /cam3, ..., /cam7 dynamically
            int camIdx = 0;
            try {
                camIdx = std::stoi(path.substr(4));
            } catch (...) {
                camIdx = 0;
            }
            serveMjpegStream(clientFd, camIdx);
        } else if (path == "/stream") {
            serveMjpegStream(clientFd, -1);
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

    void updateFrame(const std::vector<cv::Mat>& bgrImages, const WebStats& stats) {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_latestImages.clear();
        m_latestImages.reserve(bgrImages.size());
        for (const auto& img : bgrImages) {
            m_latestImages.push_back(img.clone());
        }
        m_latestStats = stats;
    }
};
