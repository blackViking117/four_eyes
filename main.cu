#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <Arena/ArenaApi.h>
#include <vector>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <queue>
#include <mutex>
#include <condition_variable>
#include <set>
#include <string>
#include <array>
#include <cstring>

#include <prepostprocessor.h>
#include <bayer_letterbox.h>

// ==============================================================================
// MODULAR WEB SERVER CONFIGURATION
// Set to 1 to enable the live MJPEG web monitor on port 8080 (http://0.0.0.0:8080)
// Set to 0 to completely disable and compile out the web server
// ==============================================================================
#define ENABLE_WEB_SERVER 1

#if ENABLE_WEB_SERVER
#include <web_streamer.hpp>
#endif

std::atomic<bool> g_running(true);
std::atomic<uint64_t> g_packetDrops(0);
std::atomic<float> g_latestCapFps(0.0f);
std::atomic<float> g_latestCapWorkMs(0.0f);
using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TRT]: " << msg << std::endl;
        }
    }
} logger;

const std::set<std::string> kRotate180Serials = {"251400510", "251102016"};

struct LetterboxInfo {
    float scale;
    int padX;
    int padY;
    int origWid = 1920;
    int origHeig = 1200;
};

// Synchronized packet containing images from all connected cameras
struct FrameBundle {
    int frameId;
    int64_t ptpTimeStampNs;
    std::vector<cv::Mat> images;
};

// Structures to accumulate run statistics for Ctrl+C summary
struct CaptureStats {
    int totalFrames = 0;
    double sumWorkMs = 0.0;
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
};

struct InferenceStats {
    int totalFrames = 0;
    double sumPreMs = 0.0;
    double sumTrtMs = 0.0;
    double sumTotalMs = 0.0;
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
};


static const std::array<std::string, 80> kClassNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

class ThreadSafeQueue {
private:
    std::queue<FrameBundle> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    size_t m_maxCapacity;

public:
    explicit ThreadSafeQueue(size_t maxCapacity = 1) : m_maxCapacity(maxCapacity) {}

    void push(FrameBundle bundle) {
        std::lock_guard<std::mutex> lock(m_mutex);

        while (m_queue.size() >= m_maxCapacity) {
            m_queue.pop();
        }

        m_queue.push(std::move(bundle));

        m_cv.notify_one();
    }

    bool pop(FrameBundle& bundle, const std::atomic<bool> & running) {
        std::unique_lock<std::mutex> lock(m_mutex);

        m_cv.wait(lock, [this, &running] {
            return !m_queue.empty() || !running.load();
        });

        if (!running.load() && m_queue.empty()) {
            return false;
        }

        bundle = std::move(m_queue.front());
        m_queue.pop();

        return true;
    }

    void wakeUpAll() {
        m_cv.notify_all();
    }
};

void letterBoxDimCal(int width, int height, LetterboxInfo& info) {
    const int targetSize = 640;

    int originalWidth  = width;
    int originalHeight = height;

    // Preserve aspect ratio
    float scale = std::min(
        static_cast<float>(targetSize) / originalWidth,
        static_cast<float>(targetSize) / originalHeight
    );

    int newWidth  = static_cast<int>(std::round(originalWidth * scale));
    int newHeight = static_cast<int>(std::round(originalHeight * scale));

    int padX = (targetSize - newWidth) / 2;
    int padY = (targetSize - newHeight) / 2;

    // Store information needed for post-processing
    info.scale = scale;
    info.padX = padX;
    info.padY = padY;
    info.origHeig = height;
    info.origWid = width;
}

std::vector<float> imageProcessorPre(const cv::Mat& input, LetterboxInfo& info) {
    const int targetSize = 640;

    int originalWidth  = input.cols;
    int originalHeight = input.rows;

    // Preserve aspect ratio
    float scale = std::min(
        static_cast<float>(targetSize) / originalWidth,
        static_cast<float>(targetSize) / originalHeight
    );

    int newWidth  = static_cast<int>(std::round(originalWidth * scale));
    int newHeight = static_cast<int>(std::round(originalHeight * scale));

    // Resize
    cv::Mat resized;
    cv::resize(
        input,
        resized,
        cv::Size(newWidth, newHeight)
    );

    // Create padded image
    cv::Mat padded(
        targetSize,
        targetSize,
        CV_8UC3,
        cv::Scalar(114, 114, 114)
    );

    int padX = (targetSize - newWidth) / 2;
    int padY = (targetSize - newHeight) / 2;

    resized.copyTo(
        padded(cv::Rect(
            padX,
            padY,
            newWidth,
            newHeight
        ))
    );

    // BGR → RGB
    cv::cvtColor(
        padded,
        padded,
        cv::COLOR_BGR2RGB
    );

    // Store information needed for post-processing
    info.scale = scale;
    info.padX = padX;
    info.padY = padY;

    // HWC → CHW
    std::vector<float> tensor(3 * targetSize * targetSize);

    for (int y = 0; y < targetSize; ++y) {
        for (int x = 0; x < targetSize; ++x) {

            cv::Vec3b pixel = padded.at<cv::Vec3b>(y, x);

            // RGB
            tensor[
                0 * targetSize * targetSize +
                y * targetSize +
                x
            ] = pixel[0] / 255.0f;

            tensor[
                1 * targetSize * targetSize +
                y * targetSize +
                x
            ] = pixel[1] / 255.0f;

            tensor[
                2 * targetSize * targetSize +
                y * targetSize +
                x
            ] = pixel[2] / 255.0f;
        }
    }

    return tensor;
}

bool imagePreprocessorGPU(FrameBundle& bundle, uint8_t* basePtr, void* output, LetterboxInfo& info, cudaStream_t preProcess) {
    int numImages = bundle.images.size();


    const int targetSize = 640;

    int originalWidth  = info.origWid;
    int originalHeight = info.origHeig;

    // Preserve aspect ratio
    float scale = info.scale;

    int newWidth  = static_cast<int>(std::round(originalWidth * scale));
    int newHeight = static_cast<int>(std::round(originalHeight * scale));

    cv::Mat inImgs[numImages];

    // PASS 1: CPU prepares all 4 images into managed memory first
    for (int i = 0; i < numImages; i++) {
        uint8_t* curr = basePtr + (i * (targetSize * targetSize * 3) * sizeof(uint8_t));
        inImgs[i] = cv::Mat(targetSize, targetSize, CV_8UC3, curr, targetSize * 3);
        inImgs[i].setTo(cv::Scalar(114, 114, 114));
        cv::Mat temp(info.origHeig, info.origWid, CV_8UC3);
        cv::cvtColor(bundle.images[i], temp, cv::COLOR_BayerBG2BGR);
        cv::Mat resized;
        cv::resize(temp, resized, cv::Size(newWidth, newHeight));
        resized.copyTo(inImgs[i](cv::Rect(info.padX, info.padY, newWidth, newHeight)));
    }


    // PASS 2: GPU launches all kernels after CPU memory writes are completely finished
    for (int i = 0; i < numImages; i++) {
        uint8_t* curr = basePtr + (i * (targetSize * targetSize * 3) * sizeof(uint8_t));
        float* batchOutput = static_cast<float*>(output) + (i * 3 * targetSize * targetSize);
        imagePreprocessor_v4<<<25, 1024, 0, preProcess>>>(curr, batchOutput);
    }

    cudaStreamSynchronize(preProcess); 
    return true;
}

bool imagePreprocessorGPU_v2(FrameBundle& bundle, uint8_t* basePtr, void* output, LetterboxInfo& info, cudaStream_t preProcess) {
    int numImages = bundle.images.size();

    const int targetSize = 640;

    int originalWidth  = info.origWid;
    int originalHeight = info.origHeig;

    // Preserve aspect ratio
    float scale = info.scale;

    int newWidth  = static_cast<int>(std::round(originalWidth * scale));
    int newHeight = static_cast<int>(std::round(originalHeight * scale));

    std::vector<cv::Mat> inImgs(numImages);

    // PASS 1: Multithreaded CPU preparation (one thread per camera in the batch)
    std::vector<std::thread> workers;
    workers.reserve(numImages);

    for (int i = 0; i < numImages; i++) {
        workers.emplace_back([&bundle, basePtr, targetSize, originalHeight, originalWidth, newWidth, newHeight, &info, &inImgs, i]() {
            uint8_t* curr = basePtr + (i * (targetSize * targetSize * 3) * sizeof(uint8_t));
            inImgs[i] = cv::Mat(targetSize, targetSize, CV_8UC3, curr, targetSize * 3);
            inImgs[i].setTo(cv::Scalar(114, 114, 114));

            cv::Mat temp(originalHeight, originalWidth, CV_8UC3);
            cv::cvtColor(bundle.images[i], temp, cv::COLOR_BayerBG2BGR);

            cv::Mat resized;
            cv::resize(temp, resized, cv::Size(newWidth, newHeight));

            resized.copyTo(inImgs[i](cv::Rect(info.padX, info.padY, newWidth, newHeight)));
        });
    }

    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    // PASS 2: GPU launches all kernels after CPU memory writes are completely finished
    for (int i = 0; i < numImages; i++) {
        uint8_t* curr = basePtr + (i * (targetSize * targetSize * 3) * sizeof(uint8_t));
        float* batchOutput = static_cast<float*>(output) + (i * 3 * targetSize * targetSize);
        imagePreprocessor_v4<<<25, 1024, 0, preProcess>>>(curr, batchOutput);
    }

    cudaStreamSynchronize(preProcess); 
    return true;
}

bool imagePreprocessorGPU_v3(FrameBundle& bundle, uint8_t* basePtr, void* output, LetterboxInfo& info, cudaStream_t preProcess) {
    int numImages = bundle.images.size();


    const int targetSize = 640;

    int originalWidth  = info.origWid;
    int originalHeight = info.origHeig;

    // Preserve aspect ratio
    float scale = info.scale;

    int newWidth  = static_cast<int>(std::round(originalWidth * scale));
    int newHeight = static_cast<int>(std::round(originalHeight * scale));

    cv::Mat inImgs[numImages];

    // PASS 1: CPU prepares all 4 images into managed memory first
    for (int i = 0; i < numImages; i++) {
        uint8_t* curr = basePtr + (i * (targetSize * targetSize * 3) * sizeof(uint8_t));
        float* batchOutput = static_cast<float*>(output) + (i * 3 * targetSize * targetSize);


        inImgs[i] = cv::Mat(targetSize, targetSize, CV_8UC3, curr, targetSize * 3);
        inImgs[i].setTo(cv::Scalar(114, 114, 114));


        cv::Mat temp(info.origHeig, info.origWid, CV_8UC3);
        cv::cvtColor(bundle.images[i], temp, cv::COLOR_BayerBG2BGR);

        cv::Mat resized;
        cv::resize(temp, resized, cv::Size(newWidth, newHeight));
        resized.copyTo(inImgs[i](cv::Rect(info.padX, info.padY, newWidth, newHeight)));

        imagePreprocessor_v4<<<25, 1024, 0, preProcess>>>(curr, batchOutput);
    }

    cudaStreamSynchronize(preProcess); 
    return true;
}

bool imagePreprocessorGPU_v5(FrameBundle& bundle, int batchSize,
                             uint8_t* rawPinned, uint8_t* rawDev, float* output,
                             const BayerLetterboxParams& lp, cudaStream_t stream) {
    const size_t one = static_cast<size_t>(lp.srcW) * lp.srcH;
 
    if (static_cast<int>(bundle.images.size()) < batchSize) return false;
 
    for (int i = 0; i < batchSize; ++i) {
        const cv::Mat& img = bundle.images[i];
        // guards against a camera resolution / pixel format change writing out of bounds
        if (img.type() != CV_8UC1 || !img.isContinuous() || img.total() != one) return false;
        std::memcpy(rawPinned + i * one, img.data, one);
    }
 
    cudaMemcpyAsync(rawDev, rawPinned, batchSize * one, cudaMemcpyHostToDevice, stream);
    launchBayerLetterbox(rawDev, output, batchSize, lp, stream);
 
    return cudaStreamSynchronize(stream) == cudaSuccess && cudaGetLastError() == cudaSuccess;
}

std::vector<cv::Mat> drawDetections(const float* output,
                                    int batchSize,
                                    const std::vector<cv::Mat>& bayerImages,
                                    const LetterboxInfo& info,
                                    uint32_t swapRBMask,   // bit i set -> camera i is BGGR (flipped 180 deg)
                                    int numDets = 300,
                                    float confThresh = 0.45f)
{
    const int stride = 6;  // x1, y1, x2, y2, score, class
    const int n = std::min<int>(batchSize, static_cast<int>(bayerImages.size()));
    std::vector<cv::Mat> drawn(n);

    for (int b = 0; b < n; ++b) {
        // OpenCV's Bayer names are shifted vs GenICam: camera RGGB -> BayerBG, camera BGGR -> BayerRG
        const int bayerCode = ((swapRBMask >> b) & 1u) ? cv::COLOR_BayerRG2BGR : cv::COLOR_BayerBG2BGR;
        cv::cvtColor(bayerImages[b], drawn[b], bayerCode);

        const float* det = output + static_cast<size_t>(b) * numDets * stride;

        for (int d = 0; d < numDets; ++d, det += stride) {
            const float score = det[4];
            if (score < confThresh) continue;

            // Undo letterbox: remove padding, then undo scale
            float x1 = (det[0] - info.padX) / info.scale;
            float y1 = (det[1] - info.padY) / info.scale;
            float x2 = (det[2] - info.padX) / info.scale;
            float y2 = (det[3] - info.padY) / info.scale;

            // Clamp to the original image
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(info.origWid  - 1)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(info.origWid  - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(info.origHeig - 1)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(info.origHeig - 1)));

            cv::Point p1(static_cast<int>(x1), static_cast<int>(y1));
            cv::Point p2(static_cast<int>(x2), static_cast<int>(y2));
            const cv::Scalar color(0, 255, 0);
            cv::rectangle(drawn[b], p1, p2, color, 3);

            // Class id -> name (falls back to the number if it's out of range)
            const int cls = static_cast<int>(det[5]);
            std::string name = (cls >= 0 && cls < static_cast<int>(kClassNames.size()))
                                   ? kClassNames[cls]
                                   : std::to_string(cls);

            char score_str[16];
            std::snprintf(score_str, sizeof(score_str), " %.2f", score);
            const std::string label = name + score_str;

            // Filled background behind the text so longer names stay readable
            int baseline = 0;
            const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
            const int ty = std::max(p1.y, ts.height + 6);  // keep label inside the image
            cv::rectangle(drawn[b],
                          cv::Point(p1.x, ty - ts.height - 6),
                          cv::Point(p1.x + ts.width, ty + baseline - 4),
                          color, cv::FILLED);
            cv::putText(drawn[b], label, cv::Point(p1.x, ty - 6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        }
    }
    return drawn;
}


bool preProcessChecker(float* preProcessCPU, float* preProcessorGPU) {
    size_t dataSize = 640*640*3;

    for (int i = 0; i < dataSize; i++) {
        if (std::fabs(preProcessCPU[i] - preProcessorGPU[i]) > 1e-4f) {
            return false;
        }
    }

    return true;
}

ThreadSafeQueue* g_queuePtr = nullptr;
std::mutex g_coutMutex; // Mutex to prevent capture and inference logs from interleaving

void sigintHandler(int) {
    g_running = false;
    if (g_queuePtr) {
        g_queuePtr->wakeUpAll();
    }
}



void triggerWorker(GenApi::INodeMap* arSystemTL,
                   GenApi::INodeMap* arNodeMap0) {
    // Latch once at startup (cameras are already confirmed Locked to PTP)
    Arena::ExecuteNode(arNodeMap0, "PtpDataSetLatch");
    int64_t ptpStart = Arena::GetNodeValue<int64_t>(arNodeMap0, "PtpDataSetLatchValue");
    auto tStart = std::chrono::steady_clock::now();

    const int64_t deltaNs = 35000000LL; // 35ms lookahead margin (generous, 100% safe against OS jitter)
    const int64_t framePeriodNs = 50000000LL; // 50.0 ms = Exactly 20.00 Hz
    const auto interval = std::chrono::nanoseconds(framePeriodNs);

    auto nextTick = tStart;
    int resyncCounter = 0;

    std::cout << "[Trigger Worker] Started: Initial PTP = " << ptpStart 
              << ". Pacing cameras at exact 20.0 Hz (50.0 ms period)\n";

    while (g_running) {
        nextTick += interval;

        auto now = std::chrono::steady_clock::now();
        if (now > nextTick) {
            nextTick = now;
        }

        // Relative elapsed offset guarantees targetPtp is ALWAYS exactly deltaNs ahead of real time
        int64_t elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now - tStart).count();
        int64_t targetPtp = ptpStart + elapsedNs + deltaNs;

        try {
            Arena::SetNodeValue<int64_t>(arSystemTL, "ActionCommandExecuteTime", targetPtp);
            Arena::ExecuteNode(arSystemTL, "ActionCommandFireCommand");
        } catch (...) {
            // Ignore on shutdown
        }

        // Periodically re-anchor PTP reference every 500 frames (~25s) to guarantee zero clock drift indefinitely
        if (++resyncCounter >= 500) {
            resyncCounter = 0;
            try {
                Arena::ExecuteNode(arNodeMap0, "PtpDataSetLatch");
                ptpStart = Arena::GetNodeValue<int64_t>(arNodeMap0, "PtpDataSetLatchValue");
                tStart = std::chrono::steady_clock::now();
            } catch (...) {}
        }

        std::this_thread::sleep_until(nextTick);
    }
    std::cout << "[Trigger Worker] Exiting cleanly...\n";
}

void captureWorker(int numCams,
                   const std::vector<Arena::IDevice*>& arDevice,
                   ThreadSafeQueue& queue,
                   CaptureStats& stats) {
    int frameCounter = 0;
    std::cout << "[Capture Worker] Started on thread: " << std::this_thread::get_id() << std::endl;

    auto t_prev_capture = std::chrono::high_resolution_clock::now();
    bool firstFrame = true;

    while (g_running) {
        auto t_cap_start = std::chrono::high_resolution_clock::now();
        if (firstFrame) {
            stats.startTime = t_cap_start;
            firstFrame = false;
        }

        // Pull images from all cameras (triggered in parallel by triggerWorker at 20 Hz)
        FrameBundle bundle;
        bundle.frameId = ++frameCounter;
        bundle.ptpTimeStampNs = 0;
        bundle.images.resize(numCams);

        bool fetchSuccess = true;
        for (int i = 0; i < numCams; i++) {
            try {
                Arena::IImage* lucidImage = arDevice[i]->GetImage(500);

                if (lucidImage->IsIncomplete()) {
                    g_packetDrops++;
                    std::lock_guard<std::mutex> lock(g_coutMutex);
                    std::cerr << "[PACKET DROP] Camera " << i << " frame " << frameCounter << " dropped packets!\n";
                }

                if (i == 0) {
                    bundle.ptpTimeStampNs = lucidImage->GetTimestampNs();
                }

                cv::Mat rawBayer(lucidImage->GetHeight(), lucidImage->GetWidth(), CV_8UC1, (void*)lucidImage->GetData());
                bundle.images[i] = rawBayer.clone(); // Fast copy of raw Bayer buffer

                arDevice[i]->RequeueBuffer(lucidImage);
            }
            catch (GenICam::GenericException& ge) {
                fetchSuccess = false;
                std::lock_guard<std::mutex> lock(g_coutMutex);
                std::cerr << "[CAPTURE WARN] Camera " << i << " frame " << frameCounter 
                          << " dropped due to SDK exception: " << ge.GetDescription() << std::endl;
            }
        }

        if (!fetchSuccess) {
            continue;
        }

        auto t_cap_end = std::chrono::high_resolution_clock::now();
        double cap_duration_ms = std::chrono::duration<double, std::milli>(t_cap_end - t_cap_start).count();
        double cap_interval_ms = std::chrono::duration<double, std::milli>(t_cap_end - t_prev_capture).count();
        t_prev_capture = t_cap_end;
        double cap_fps = (frameCounter > 1 && cap_interval_ms > 0.0) ? (1000.0 / cap_interval_ms) : (1000.0 / cap_duration_ms);

        stats.totalFrames++;
        stats.sumWorkMs += cap_duration_ms;
        stats.endTime = t_cap_end;

        g_latestCapFps.store(static_cast<float>(cap_fps));
        g_latestCapWorkMs.store(static_cast<float>(cap_duration_ms));

        // Push the synchronized bundle into the thread-safe queue
        queue.push(std::move(bundle));

        {
            std::lock_guard<std::mutex> lock(g_coutMutex);
            std::cout << "[CAPTURE]   Frame " << frameCounter
                      << " | Work: " << cap_duration_ms << " ms"
                      << " | Cycle: " << cap_interval_ms << " ms"
                      << " | Rate: " << cap_fps << " FPS"
                      << std::endl;
        }
    }
    std::cout << "[Capture Worker] Exiting cleanly...\n";
}



void inferenceWorker(int batchSize,
                     IExecutionContext* context,
                     const char* inputName,
                     const char* outputName,
                     void* inputs[2],
                     void* outputs[2],
                     cudaStream_t stream,
                     ThreadSafeQueue& queue,
                     InferenceStats& stats,
                     LetterboxInfo& info,
                     uint32_t swapRBMask
#if ENABLE_WEB_SERVER
                     , MJPEGWebServer* webServer = nullptr
#endif
) {
    std::cout << "[Inference Worker] Started on thread: " << std::this_thread::get_id() << std::endl;
 
    int inferCount = 0;
    auto t_prev_infer = std::chrono::high_resolution_clock::now();
 
    // Network input size (must match the engine and the tensors allocated in main)
    constexpr int NET_W = 640, NET_H = 640;
    BayerLetterboxParams lp = makeBayerLetterboxParams(info.origWid, info.origHeig, NET_W, NET_H);
    lp.swapRBMask = swapRBMask;   // cameras that deliver BGGR (flipped 180 deg) get R/B swapped in the kernel
 
    // Raw Bayer staging: CPU writes rawPinned -> H2D copy -> rawDev -> kernel reads rawDev.
    // One pair is enough because imagePreprocessorGPU_v5 blocks until the kernel has finished.
    const size_t rawBytes = static_cast<size_t>(batchSize) * lp.srcW * lp.srcH;
    uint8_t* rawPinned = nullptr;
    uint8_t* rawDev = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void**>(&rawPinned), rawBytes, cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&rawDev), rawBytes) != cudaSuccess) {
        std::cerr << "[Inference Worker] Failed to allocate raw Bayer staging buffers\n";
        if (rawPinned) cudaFreeHost(rawPinned);
        g_running = false;   // nothing can run without them; let the other threads wind down
        return;
    }

 
    // High-priority stream so the ~1 ms preprocessing kernel is not stuck behind TensorRT kernels
    int prioLeast = 0, prioGreatest = 0;
    cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest);
    cudaStream_t preProcess;
    cudaStreamCreateWithPriority(&preProcess, cudaStreamNonBlocking, prioGreatest);
 
    int currentSlot = 0;
    int nextSlot = 1;
 
    // --- PIPELINE SEEDING (Batch 0) ---
    FrameBundle currentBundle;
    if (!queue.pop(currentBundle, g_running)) {
        std::cout << "[Inference Worker] Exiting cleanly before starting...\n";
        cudaStreamDestroy(preProcess);
        cudaFreeHost(rawPinned);
        cudaFree(rawDev);
        return;
    }
 
    inferCount++;
    auto t_work_start = std::chrono::high_resolution_clock::now();
    stats.startTime = t_work_start;
 
    // Preprocess Batch 0 into Slot 0
    auto t_seed_pre_start = std::chrono::high_resolution_clock::now();
    if (!imagePreprocessorGPU_v5(currentBundle, batchSize, rawPinned, rawDev,
                                 static_cast<float*>(inputs[currentSlot]), lp, preProcess)) {
        std::cout << "GPU preprocessing FAILED on seed batch!\n" << std::flush;
    }

    auto t_seed_pre_end = std::chrono::high_resolution_clock::now();
    double last_pre_ms = std::chrono::duration<double, std::milli>(t_seed_pre_end - t_seed_pre_start).count();
 
    // Launch TensorRT on Batch 0 asynchronously
    auto t_trt_start = std::chrono::high_resolution_clock::now();
    context->setTensorAddress(inputName, inputs[currentSlot]);
    context->setTensorAddress(outputName, outputs[currentSlot]);
    context->enqueueV3(stream);
 
    // --- STEADY-STATE DOUBLE-BUFFERED PIPELINE LOOP ---
    while (g_running) {
        FrameBundle nextBundle;
        // 1. Pop next bundle from queue
        if (!queue.pop(nextBundle, g_running)) {
            break; // Ctrl+C signaled shutdown
        }
 
        // 2. OVERLAP: While GPU is busy inferring currentSlot on `stream`,
        //             the preProcess stream builds the tensor for nextBundle into nextSlot!
        auto t_pre_start = std::chrono::high_resolution_clock::now();
        if (!imagePreprocessorGPU_v5(nextBundle, batchSize, rawPinned, rawDev,
                                     static_cast<float*>(inputs[nextSlot]), lp, preProcess)) {
            std::cout << "GPU preprocessing FAILED on batch " << nextBundle.frameId << "!\n" << std::flush;
        }
        auto t_pre_end = std::chrono::high_resolution_clock::now();
        double pre_ms = std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();
 
        // 3. Wait for currentSlot's TensorRT inference to finish on `stream`
        cudaStreamSynchronize(stream);
        auto t_trt_end = std::chrono::high_resolution_clock::now();
        double trt_ms = std::chrono::duration<double, std::milli>(t_trt_end - t_trt_start).count();
 
        auto t_work_end = t_trt_end;
        double total_work_ms = std::chrono::duration<double, std::milli>(t_work_end - t_work_start).count();
        double infer_interval_ms = std::chrono::duration<double, std::milli>(t_work_end - t_prev_infer).count();
        t_prev_infer = t_work_end;
        double infer_fps = (inferCount > 1 && infer_interval_ms > 0.0) ? (1000.0 / infer_interval_ms) : (1000.0 / total_work_ms);
 
        // 4. IMMEDIATELY launch TensorRT on the nextSlot (already preprocessed in background!)
        //    This keeps the GPU 100% saturated with ZERO idle wait time.
        inferCount++;
        t_work_start = std::chrono::high_resolution_clock::now();
        t_trt_start = t_work_start;
 
        context->setTensorAddress(inputName, inputs[nextSlot]);
        context->setTensorAddress(outputName, outputs[nextSlot]);
        context->enqueueV3(stream);

        stats.totalFrames++;
        stats.sumPreMs += last_pre_ms;
        stats.sumTrtMs += trt_ms;
        stats.sumTotalMs += total_work_ms;
        stats.endTime = t_work_end;

        std::vector<cv::Mat> results = drawDetections(static_cast<float*>(outputs[currentSlot]), batchSize, currentBundle.images, info, swapRBMask);

#if ENABLE_WEB_SERVER
        if (webServer) {
            WebStats wStats;
            wStats.frameId = currentBundle.frameId;
            wStats.capFps = g_latestCapFps.load();
            wStats.infFps = infer_fps;
            wStats.capWorkMs = g_latestCapWorkMs.load();
            wStats.trtMs = trt_ms;
            wStats.packetDrops = g_packetDrops.load();
            wStats.numCams = batchSize;
            webServer->updateFrames(currentBundle.images, results, wStats);
        }
#endif
 
        {
            std::lock_guard<std::mutex> lock(g_coutMutex);
            std::cout << "[INFERENCE] Frame " << currentBundle.frameId
                      << " | Pre: " << last_pre_ms << " ms"
                      << " | TRT: " << trt_ms << " ms"
                      << " | Total: " << total_work_ms << " ms"
                      << " | Rate: " << infer_fps << " FPS"
                      << std::endl;
        }
 
        // 7. Ping-pong swap slots and hand over bundle
        last_pre_ms = pre_ms;
        currentBundle = std::move(nextBundle);
        std::swap(currentSlot, nextSlot);
    }
 
    // Drain the final launched batch
    cudaStreamSynchronize(stream);
    auto t_drain_end = std::chrono::high_resolution_clock::now();
    double final_trt_ms = std::chrono::duration<double, std::milli>(t_drain_end - t_trt_start).count();
    double final_total_ms = std::chrono::duration<double, std::milli>(t_drain_end - t_work_start).count();
    double final_interval_ms = std::chrono::duration<double, std::milli>(t_drain_end - t_prev_infer).count();
    double final_fps = (inferCount > 1 && final_interval_ms > 0.0) ? (1000.0 / final_interval_ms) : (1000.0 / final_total_ms);
 
    stats.totalFrames++;
    stats.sumPreMs += last_pre_ms;
    stats.sumTrtMs += final_trt_ms;
    stats.sumTotalMs += final_total_ms;
    stats.endTime = t_drain_end;

    std::vector<cv::Mat> finalResults = drawDetections(static_cast<float*>(outputs[currentSlot]), batchSize, currentBundle.images, info, swapRBMask);

#if ENABLE_WEB_SERVER
    if (webServer) {
        WebStats wStats;
        wStats.frameId = currentBundle.frameId;
        wStats.capFps = g_latestCapFps.load();
        wStats.infFps = final_fps;
        wStats.capWorkMs = g_latestCapWorkMs.load();
        wStats.trtMs = final_trt_ms;
        wStats.packetDrops = g_packetDrops.load();
        wStats.numCams = batchSize;
        webServer->updateFrames(currentBundle.images, finalResults, wStats);
    }
#endif
 
    {
        std::lock_guard<std::mutex> lock(g_coutMutex);
        std::cout << "[INFERENCE] Frame " << currentBundle.frameId
                  << " | Pre: " << last_pre_ms << " ms"
                  << " | TRT: " << final_trt_ms << " ms"
                  << " | Total: " << final_total_ms << " ms"
                  << " | Rate: " << final_fps << " FPS"
                  << std::endl;
    }
 
    std::cout << "[Inference Worker] Exiting cleanly...\n";
    cudaStreamDestroy(preProcess);
    cudaFreeHost(rawPinned);
    cudaFree(rawDev);
}



int main() {
    std::string engineFilePath = "/media/swdep/500GB_SSD_PERCEP/black_magic/yolov10x_dynamic_4_fp16.engine";

    std::ifstream engineFile(engineFilePath, std::ios::binary | std::ios::ate);
    if (!engineFile) {
        std::cout << "Couldn't Open the Engine File at: " << engineFilePath << std::endl;
        return -1;
    }
    
    size_t engineSize = static_cast<size_t>(engineFile.tellg());
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);

    if (!engine) {
        std::cout << "Engine deserialization Failure." << std::endl;
        return -1;
    }

    IExecutionContext *context = engine->createExecutionContext();
    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);

    Arena::ISystem* arSystem = nullptr;

    try {
        arSystem = Arena::OpenSystem();
        
        arSystem->UpdateDevices(100);
        std::vector<Arena::DeviceInfo> cameras = arSystem->GetDevices();

        if (cameras.empty()) {
            std::cout << "No LUCID camera found!!!!" << std::endl;
            Arena::CloseSystem(arSystem);
            return -1;
        }

        int numCams = cameras.size();
        std::cout << "Found: " << numCams << " number of cameras.\n";

        // ==============================================================
        // DYNAMIC BATCH SIZING BASED ON CONNECTED CAMERAS
        // ==============================================================
        constexpr int MAX_VEHICLE_CAMERAS = 8;
        int batchSize = numCams;

        if (batchSize > MAX_VEHICLE_CAMERAS) {
            std::cout << "[WARN] Found " << numCams << " cameras, which exceeds MAX_VEHICLE_CAMERAS (" 
                      << MAX_VEHICLE_CAMERAS << "). Clamping batch size to " << MAX_VEHICLE_CAMERAS << ".\n";
            batchSize = MAX_VEHICLE_CAMERAS;
        }

        std::cout << "[TensorRT] Dynamic batch sizing: setting batch size to " << batchSize << "\n";

        // Set dynamic batch input shape (BatchSize x 3 x 640 x 640)
        if (!context->setInputShape(inputName, Dims4{batchSize, 3, 640, 640})) {
            std::cerr << "[ERROR] Failed to set input shape for batch size " << batchSize 
                      << ". Check if the engine supports this batch profile." << std::endl;
            return -1;
        }

        // Output tensor shape automatically scales with the dynamic batch size
        Dims const outputDims = context->getTensorShape(outputName);
        size_t outputCount = 1;
        for (int32_t i = 0; i < outputDims.nbDims; ++i) {
            outputCount *= outputDims.d[i];
        }

        std::cout << "[TensorRT] Output tensor dims: [";
        for (int32_t i = 0; i < outputDims.nbDims; ++i) {
            std::cout << outputDims.d[i] << (i < outputDims.nbDims - 1 ? ", " : "");
        }
        std::cout << "] (Total elements: " << outputCount << ")\n";

        // Allocate double-buffered host-pinned memory scaled for the dynamic batch size
        void* inputs[2] = {nullptr, nullptr};
        void* outputs[2] = {nullptr, nullptr};
        size_t inputSize = static_cast<size_t>(batchSize) * 3 * 640 * 640 * sizeof(float);
        size_t outputSize = outputCount * sizeof(float);

        for (int i = 0; i < 2; i++) {
            cudaHostAlloc(&inputs[i],  inputSize,  cudaHostAllocMapped);
            cudaHostAlloc(&outputs[i], outputSize, cudaHostAllocMapped);
        }

        context->setTensorAddress(inputName, inputs[0]);
        context->setTensorAddress(outputName, outputs[0]);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        std::vector<Arena::IDevice*> arDevice(numCams, nullptr);
        std::vector<GenApi::INodeMap*> arNodeMap(numCams, nullptr);

        // Bit i set -> camera i delivers BGGR (flipped 180 deg) instead of RGGB. Filled from the
        // PixelFormat each camera reports after the flip, so it stays correct if the serial list changes.
        uint32_t swapRBMask = 0;

        for (int i = 0; i < numCams; i++) {
            arDevice[i] = arSystem->CreateDevice(cameras[i]);
            arNodeMap[i] = arDevice[i]->GetNodeMap();
            auto* arStreamNodeMap = arDevice[i]->GetTLStreamNodeMap();

            //configuring each camera

            //pixel formatting
            const std::string serial = cameras[i].SerialNumber().c_str();
            const bool rotate180 = kRotate180Serials.count(serial) > 0;

            // 1. Clear any flip left over from a previous run
            Arena::SetNodeValue<bool>(arNodeMap[i], "ReverseX", false);
            Arena::SetNodeValue<bool>(arNodeMap[i], "ReverseY", false);

            // 2. With no flip active, BayerRG8 is writable
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "PixelFormat", "BayerRG8");

            // 3. Apply the flip; the camera updates PixelFormat itself
            Arena::SetNodeValue<bool>(arNodeMap[i], "ReverseX", rotate180);
            Arena::SetNodeValue<bool>(arNodeMap[i], "ReverseY", rotate180);

            GenICam::gcstring pf = Arena::GetNodeValue<GenICam::gcstring>(arNodeMap[i], "PixelFormat");
            std::cout << "  Camera [" << i << "] " << serial
                    << " | rotate180=" << rotate180
                    << " | PixelFormat=" << pf << "\n" << std::flush;

            if (pf == "BayerBG8") {
                swapRBMask |= (1u << i);
            } else if (pf != "BayerRG8") {
                std::cerr << "[WARN] Camera " << i << " reports " << pf
                          << " - kernel only handles BayerRG8 / BayerBG8\n";
            }

            //exposure set to equal time for each camera
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "ExposureAuto", "Off");
            Arena::SetNodeValue<double>(arNodeMap[i], "ExposureTime", 20000.0);
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "GainAuto", "Off");
            Arena::SetNodeValue<double>(arNodeMap[i], "Gain", 10.0);

            //Trigger configuration
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "TriggerSelector", "FrameStart");
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "TriggerMode", "On");
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "TriggerSource", "Action0");
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "TriggerOverlap", "PreviousFrame"); // Enable overlapped readout & exposure to break the 14 FPS barrier

            //acttion configuration
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "ActionUnconditionalMode", "On");
            Arena::SetNodeValue<int64_t>(arNodeMap[i], "ActionSelector", 0);
            Arena::SetNodeValue<int64_t>(arNodeMap[i], "ActionDeviceKey", 1);
            Arena::SetNodeValue<int64_t>(arNodeMap[i], "ActionGroupKey", 1);
            Arena::SetNodeValue<int64_t>(arNodeMap[i], "ActionGroupMask", 1);

            //Setting all cameras to slave
            Arena::SetNodeValue<bool>(arNodeMap[i], "PtpSlaveOnly", true);
            if (!Arena::GetNodeValue<bool>(arNodeMap[i], "PtpEnable")) {
                Arena::SetNodeValue<bool>(arNodeMap[i], "PtpEnable", true);
            }

            // Inter-packet delay (GevSCPD): on 10 GbE uplink, 0 ticks unlocks full 1 Gbps wire speed (~18.5ms per camera)
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "TransferControlMode", "Basic");
            Arena::SetNodeValue<int64_t>(arNodeMap[i], "GevSCPD", 0);

            //stream configuration
            Arena::SetNodeValue<bool>(arStreamNodeMap, "StreamAutoNegotiatePacketSize", true);
            Arena::SetNodeValue<bool>(arStreamNodeMap, "StreamPacketResendEnable", true);

            //pixel configuration
            Arena::SetNodeValue<int64_t>(arDevice[i]->GetNodeMap(), "Width", 1920);
            Arena::SetNodeValue<int64_t>(arDevice[i]->GetNodeMap(), "Height", 1200);
            Arena::SetNodeValue<int64_t>(arDevice[i]->GetNodeMap(), "OffsetX", 0);
            Arena::SetNodeValue<int64_t>(arDevice[i]->GetNodeMap(), "OffsetY", 0);
        }

        LetterboxInfo info;
        letterBoxDimCal(1920, 1200, info);


        // Configure System Action Command Broadcast
        GenApi::INodeMap* arSystemTL = arSystem->GetTLSystemNodeMap();
        Arena::SetNodeValue<int64_t>(arSystemTL, "ActionCommandDeviceKey", 1);
        Arena::SetNodeValue<int64_t>(arSystemTL, "ActionCommandGroupKey", 1);
        Arena::SetNodeValue<int64_t>(arSystemTL, "ActionCommandGroupMask", 1);
        Arena::SetNodeValue<int64_t>(arSystemTL, "ActionCommandTargetIP", 0xFFFFFFFF); // 0xFFFFFFFF = Broadcast
        
        bool alllocked = false;
        while(!alllocked) {
            int lockedNum = 0;
            for (int i = 0; i < numCams; i++) {
                GenICam::gcstring ptpStatus = Arena::GetNodeValue<GenICam::gcstring>(arNodeMap[i], "PtpStatus");
                GenICam::gcstring servoStatus = Arena::GetNodeValue<GenICam::gcstring>(arNodeMap[i], "PtpServoStatus");

                if (ptpStatus == "Slave" && servoStatus == "Locked") {
                    lockedNum++;
                }
            }

            if (lockedNum == numCams) {
                alllocked = true;
            } else {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                std::cout << "." << std::flush;
            }
        }

        std::signal(SIGINT, sigintHandler);

        for (int i = 0; i < numCams; i++) {
            arDevice[i]->StartStream();
        }

        ThreadSafeQueue frameQueue(1);
        g_queuePtr = &frameQueue; // Connect signal handler

        CaptureStats capStats;
        InferenceStats infStats;

#if ENABLE_WEB_SERVER
        MJPEGWebServer webServer(8080);
        if (!webServer.start()) {
            std::cerr << "[WARN] Failed to start Web Server on port 8080\n";
        }
#endif

        std::cout << "\n[PIPELINE] Launching Trigger (20Hz), Capture, and Inference threads. Press Ctrl+C to stop...\n\n";

        // Dedicated high-precision 20.0 Hz PTP Action Command generator
        std::thread triggerThread(triggerWorker, 
                                 arSystemTL, 
                                 arNodeMap[0]);

        // Capture worker ingesting frames as they arrive from all 4 cameras
        std::thread cameraThread(captureWorker, 
                                 numCams, 
                                 std::cref(arDevice), 
                                 std::ref(frameQueue),
                                 std::ref(capStats));

        std::thread inferenceThread(inferenceWorker, 
                                  batchSize, 
                                  context, 
                                  inputName,
                                  outputName,
                                  inputs, 
                                  outputs, 
                                  stream, 
                                  std::ref(frameQueue),
                                  std::ref(infStats),
                                  std::ref(info),
                                  swapRBMask
#if ENABLE_WEB_SERVER
                                  , &webServer
#endif
        );

        cameraThread.join();
        triggerThread.join();
        inferenceThread.join();
        g_queuePtr = nullptr;

#if ENABLE_WEB_SERVER
        webServer.stop();
#endif

        std::cout << "\n[PIPELINE] Both threads terminated. Releasing resources...\n";

        // Print final performance averages across entire run
        std::cout << "\n============================================================\n";
        std::cout << "               PIPELINE BENCHMARK SUMMARY                   \n";
        std::cout << "============================================================\n";

        if (capStats.totalFrames > 1) {
            double capWallSec = std::chrono::duration<double>(capStats.endTime - capStats.startTime).count();
            double avgCapRate = (capStats.totalFrames - 1) / capWallSec;
            double avgCapWorkMs = capStats.sumWorkMs / capStats.totalFrames;

            std::cout << "[CAPTURE SUMMARY]:\n"
                      << "  Total Batches Captured : " << capStats.totalFrames << "\n"
                      << "  Total Images Ingested  : " << capStats.totalFrames * numCams << " (" << numCams << " cameras)\n"
                      << "  Avg Capture Latency    : " << avgCapWorkMs << " ms\n"
                      << "  Avg Capture Rate       : " << avgCapRate << " FPS per camera\n"
                      << "  Avg Ingestion Rate     : " << avgCapRate * numCams << " images/sec\n\n";
        } else {
            std::cout << "[CAPTURE SUMMARY]: Not enough frames captured to compute average.\n\n";
        }

        if (infStats.totalFrames > 1) {
            double infWallSec = std::chrono::duration<double>(infStats.endTime - infStats.startTime).count();
            double avgInfRate = (infStats.totalFrames - 1) / infWallSec;
            double avgPreMs = infStats.sumPreMs / infStats.totalFrames;
            double avgTrtMs = infStats.sumTrtMs / infStats.totalFrames;
            double avgTotalMs = infStats.sumTotalMs / infStats.totalFrames;

            std::cout << "[INFERENCE SUMMARY]:\n"
                      << "  Total Batches Inferred : " << infStats.totalFrames << "\n"
                      << "  Total Images Processed : " << infStats.totalFrames * batchSize << " (Batch Size: " << batchSize << ")\n"
                      << "  Avg Preprocessing Time : " << avgPreMs << " ms\n"
                      << "  Avg TensorRT Time      : " << avgTrtMs << " ms\n"
                      << "  Avg Total Latency      : " << avgTotalMs << " ms\n"
                      << "  Avg Inference Rate     : " << avgInfRate << " FPS per camera\n"
                      << "  Avg Total Throughput   : " << avgInfRate * batchSize << " images/sec\n";
        } else {
            std::cout << "[INFERENCE SUMMARY]: Not enough frames inferred to compute average.\n";
        }

        std::cout << "\n[NETWORK & PACKET INTEGRITY SUMMARY]:\n"
                  << "  Total Incomplete / Dropped Buffers : " << g_packetDrops.load() << "\n";
        if (capStats.totalFrames > 0) {
            double dropPct = ((double)g_packetDrops.load() / (capStats.totalFrames * numCams)) * 100.0;
            std::cout << "  Packet Drop Rate                   : " << dropPct << " %\n";
            if (g_packetDrops.load() == 0) {
                std::cout << "  Network Stream Integrity           : PERFECT (0 packet drops detected)\n";
            } else {
                std::cout << "  Network Stream Integrity           : DEGRADED (" << g_packetDrops.load() << " incomplete frames)\n";
            }
        }
        std::cout << "============================================================\n\n";

        // Clean up CUDA stream and pinned host memory
        cudaStreamDestroy(stream);
        for (int i = 0; i < 2; i++) {
            cudaFreeHost(inputs[i]);
            cudaFreeHost(outputs[i]);
        }

        delete context;
        delete engine;
        delete runtime;

        for (int i = 0; i < numCams; i++) {
            arDevice[i]->StopStream();
            arSystem->DestroyDevice(arDevice[i]);
        }
        Arena::CloseSystem(arSystem);
    }
    catch (GenICam::GenericException& ge)
    {
        // GenICam exceptions handle SDK-specific errors (e.g., network drops, invalid nodes)
        std::cerr << "Arena SDK Exception: " << ge.GetDescription() << std::endl;
        
        // Ensure cleanup occurs even if an exception is thrown
        if (arSystem) Arena::CloseSystem(arSystem);
        
        return -1;
    }
    catch (std::exception& ex)
    {
        std::cerr << "Standard Exception: " << ex.what() << std::endl;
        return -1;
    }

    

    return 0;
}