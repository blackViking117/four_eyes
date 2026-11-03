#include <iostream>
#include <iomanip>
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
#include <map>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <mask_codec.hpp>
#include <mask_builder.hpp>
#include <pipeline_config.hpp>

#include <prepostprocessor.h>
#include <bayer_letterbox.h>
#include <byte_tracker.hpp>

// ==============================================================================
// MODULAR WEB SERVER CONFIGURATION
// Set to 1 to enable the live MJPEG web monitor on port 8080 (http://0.0.0.0:8080)
// Set to 0 to completely disable and compile out the web server
// ==============================================================================
#define ENABLE_WEB_SERVER 1

#if ENABLE_WEB_SERVER
#include <web_streamer.hpp>
#endif

// ==============================================================================
// 2D MULTI-OBJECT TRACKING (ByteTrack-style, per camera, see include/byte_tracker.hpp)
// Runtime tracking is controlled via config.yaml (tracking.enabled: true / false).
// ==============================================================================
#define ENABLE_TRACKING 1

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
    double sumPostMs = 0.0;
    double sumTotalMs = 0.0;
    uint64_t sumTargets = 0;
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

// Container for frames, detections, and masks to be saved to disk
struct SavedFrameBundle {
    enum Type { PRE_CAPTURE, INFERENCE_RESULT } type = PRE_CAPTURE;
    int frameId = 0;
    int64_t ptpTimeStampNs = 0;
    std::vector<std::string> cameraSerials;
    uint32_t swapRBMask = 0;

    // For PRE_CAPTURE (from captureWorker directly at full camera rate):
    std::vector<cv::Mat> rawBayerImages;

    // For INFERENCE_RESULT (from inferenceWorker):
    std::vector<cv::Mat> detectionImages;  // images with bounding box overlays
    std::vector<cv::Mat> maskImages;       // individual camera masks
    cv::Mat combinedMask;                  // horizontal combined mask
};

struct SaveStats {
    int totalBundlesSaved = 0;
    int totalImagesSaved = 0;
    double totalSaveTimeMs = 0.0;
};

class ThreadSafeSaveQueue {
private:
    std::queue<SavedFrameBundle> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cvNotEmpty;
    std::condition_variable m_cvNotFull;
    size_t m_maxCapacity;
    std::atomic<bool> m_shutdown{false};
    std::atomic<uint64_t> m_droppedCount{0};

public:
    explicit ThreadSafeSaveQueue(size_t maxCapacity = 500) : m_maxCapacity(maxCapacity) {}

    bool push(SavedFrameBundle bundle, int waitMs = 20) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_maxCapacity > 0 && m_queue.size() >= m_maxCapacity) {
            m_cvNotFull.wait_for(lock, std::chrono::milliseconds(waitMs), [this] {
                return m_queue.size() < m_maxCapacity || m_shutdown.load();
            });
        }
        if (m_shutdown.load()) return false;
        if (m_maxCapacity > 0 && m_queue.size() >= m_maxCapacity) {
            m_queue.pop();
            m_droppedCount++;
            std::cerr << "[FRAME SAVER WARN] Queue full (" << m_maxCapacity 
                      << ")! Dropped oldest bundle. Total dropped: " << m_droppedCount.load() << "\n";
        }
        m_queue.push(std::move(bundle));
        m_cvNotEmpty.notify_one();
        return true;
    }

    bool pop(SavedFrameBundle& bundle, const std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cvNotEmpty.wait(lock, [this, &running] {
            return !m_queue.empty() || (!running.load() && m_queue.empty()) || m_shutdown.load();
        });

        if (m_queue.empty()) {
            return false;
        }

        bundle = std::move(m_queue.front());
        m_queue.pop();
        m_cvNotFull.notify_one();
        return true;
    }

    void shutdown() {
        m_shutdown.store(true);
        m_cvNotEmpty.notify_all();
        m_cvNotFull.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    uint64_t getDroppedCount() const {
        return m_droppedCount.load();
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
                                    cv::Mat& combinedMask,
                                    const MaskOptions& maskOpt,
                                    int numDets = 300,
                                    float confThresh = 0.45f,
                                    int* outTotalDets = nullptr,
                                    std::vector<cv::Mat>* outPreImages = nullptr)
{
    if (outTotalDets) *outTotalDets = 0;
    const int stride = 6;  // x1, y1, x2, y2, score, class
    const int n = std::min<int>(batchSize, static_cast<int>(bayerImages.size()));
    std::vector<cv::Mat> drawn(n);

    const int W = info.origWid;
    const int H = info.origHeig;
    const int maskType = maskOpt.single_channel ? CV_8UC1 : CV_8UC3;

    // Allocate horizontal combined mask canvas: Height H x Width (n * W)
    combinedMask = cv::Mat::zeros(H, n * W, maskType);

    for (int b = 0; b < n; ++b) {
        // OpenCV's Bayer names are shifted vs GenICam: camera RGGB -> BayerBG, camera BGGR -> BayerRG
        const int bayerCode = ((swapRBMask >> b) & 1u) ? cv::COLOR_BayerRG2BGR : cv::COLOR_BayerBG2BGR;
        cv::cvtColor(bayerImages[b], drawn[b], bayerCode);
        if (outPreImages && b < static_cast<int>(outPreImages->size())) {
            (*outPreImages)[b] = drawn[b].clone();
        }

        // Sub-region ROI in combinedMask for camera b (horizontal layout: [b*W .. (b+1)*W))
        cv::Mat subMask = combinedMask(cv::Rect(b * W, 0, W, H));
        cv::Mat gray;
        if (maskOpt.mode != MaskMode::BBOX) {
            cv::cvtColor(drawn[b], gray, cv::COLOR_BGR2GRAY);
        }

        const float* det = output + static_cast<size_t>(b) * numDets * stride;
        int instCounter = 0;

        for (int d = 0; d < numDets; ++d, det += stride) {
            const float score = det[4];
            if (score < confThresh) continue;
            if (outTotalDets) (*outTotalDets)++;

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

            // Paint mask with mono8 codec: bits[7:3] instance id (1..31), bits[2:0] class id
            int inst_id = std::min(++instCounter, maskcodec::MAX_INST);
            cv::Rect box(p1, p2);
            paintDetection(subMask, gray, box, inst_id, cls, maskOpt);
        }
    }
    return drawn;
}

// Tracking variant of drawDetections(): decodes the raw YOLOv10 output into original-image boxes,
// advances the per-camera ByteTrack by one frame, then draws confirmed tracks on display images
// and paints each detected object into the combined horizontal mask using the mask codec.
std::vector<cv::Mat> trackAndDraw(const float* output,
                                  int batchSize,
                                  const std::vector<cv::Mat>& bayerImages,
                                  const LetterboxInfo& info,
                                  uint32_t swapRBMask,   // bit i set -> camera i is BGGR (flipped 180 deg)
                                  tracker2d::MultiCamByteTracker& tracker,
                                  std::vector<tracker2d::Detection>& dets,   // scratch, reused every call
                                  cv::Mat& combinedMask,
                                  const MaskOptions& maskOpt,
                                  int numDets = 300,
                                  int* outTotalTracks = nullptr,
                                  std::vector<cv::Mat>* outPreImages = nullptr)
{
    if (outTotalTracks) *outTotalTracks = 0;
    const int stride = 6;  // x1, y1, x2, y2, score, class
    const int n = std::min<int>(batchSize, static_cast<int>(bayerImages.size()));
    const float minScore = tracker.config().low_thresh;   // the tracker ignores anything below this anyway
    std::vector<cv::Mat> drawn(n);

    const int W = info.origWid;
    const int H = info.origHeig;
    const int maskType = maskOpt.single_channel ? CV_8UC1 : CV_8UC3;

    // Allocate horizontal combined mask canvas: Height H x Width (n * W)
    combinedMask = cv::Mat::zeros(H, n * W, maskType);

    for (int b = 0; b < n; ++b) {
        // 1. Raw output -> original-image boxes (same letterbox undo + clamp as drawDetections)
        dets.clear();
        const float* det = output + static_cast<size_t>(b) * numDets * stride;
        for (int d = 0; d < numDets; ++d, det += stride) {
            const float score = det[4];
            if (score < minScore) continue;

            float x1 = (det[0] - info.padX) / info.scale;
            float y1 = (det[1] - info.padY) / info.scale;
            float x2 = (det[2] - info.padX) / info.scale;
            float y2 = (det[3] - info.padY) / info.scale;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(info.origWid  - 1)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(info.origWid  - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(info.origHeig - 1)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(info.origHeig - 1)));

            const float w = x2 - x1;
            const float h = y2 - y1;
            if (w < 1.0f || h < 1.0f) continue;   // degenerate after clamping

            dets.push_back({cv::Rect2f(x1, y1, w, h), score, static_cast<int>(det[5])});
        }

        // 2. Tracking (high/low split, GIoU + IoU association, Kalman) for camera b
        tracker.update(b, dets);

        // 3. Debayer for display (OpenCV's Bayer names are shifted vs GenICam, see drawDetections)
        const int bayerCode = ((swapRBMask >> b) & 1u) ? cv::COLOR_BayerRG2BGR : cv::COLOR_BayerBG2BGR;
        cv::cvtColor(bayerImages[b], drawn[b], bayerCode);
        if (outPreImages && b < static_cast<int>(outPreImages->size())) {
            (*outPreImages)[b] = drawn[b].clone();
        }

        // 4. Sub-region ROI in combinedMask for camera b (horizontal layout: [b*W .. (b+1)*W))
        cv::Mat subMask = combinedMask(cv::Rect(b * W, 0, W, H));
        cv::Mat gray;
        if (maskOpt.mode != MaskMode::BBOX) {
            cv::cvtColor(drawn[b], gray, cv::COLOR_BGR2GRAY);
        }

        // 5. Draw confirmed tracks on display image AND paint into mask
        const cv::Rect imgRect(0, 0, drawn[b].cols, drawn[b].rows);
        const cv::Scalar color(0, 255, 0);
        std::set<int> usedInstIds;

        for (const auto& t : tracker.tracks(b)) {
            if (!t.isReportable()) continue;

            cv::Rect box(cvRound(t.bbox.x), cvRound(t.bbox.y), cvRound(t.bbox.width), cvRound(t.bbox.height));
            box &= imgRect;
            if (box.width <= 0 || box.height <= 0) continue;
            if (outTotalTracks) (*outTotalTracks)++;
            cv::rectangle(drawn[b], box, color, 3);

            const int cls = t.label;
            const std::string name = (cls >= 0 && cls < static_cast<int>(kClassNames.size()))
                                         ? kClassNames[cls]
                                         : std::to_string(cls);
            char score_str[16];
            std::snprintf(score_str, sizeof(score_str), " %.2f", t.score);
            const std::string label = "ID:" + std::to_string(t.id) + " " + name + score_str;

            int baseline = 0;
            const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
            const int ty = std::max(box.y, ts.height + 6);  // keep label inside the image
            cv::rectangle(drawn[b],
                          cv::Point(box.x, ty - ts.height - 6),
                          cv::Point(box.x + ts.width, ty + baseline - 4),
                          color, cv::FILLED);
            cv::putText(drawn[b], label, cv::Point(box.x, ty - 6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

            // Paint mask with mono8 codec: bits[7:3] instance id (1..31), bits[2:0] class id
            int inst_id = (t.id > 0) ? (((t.id - 1) % maskcodec::MAX_INST) + 1) : 1;
            while (usedInstIds.count(inst_id) && usedInstIds.size() < static_cast<size_t>(maskcodec::MAX_INST)) {
                inst_id = (inst_id % maskcodec::MAX_INST) + 1;
            }
            usedInstIds.insert(inst_id);

            paintDetection(subMask, gray, box, inst_id, cls, maskOpt);
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
ThreadSafeSaveQueue* g_saveQueuePtr = nullptr;
std::mutex g_coutMutex; // Mutex to prevent capture and inference logs from interleaving

void sigintHandler(int) {
    g_running = false;
    if (g_queuePtr) {
        g_queuePtr->wakeUpAll();
    }
    if (g_saveQueuePtr) {
        g_saveQueuePtr->shutdown();
    }
    if (ros::isInitialized()) {
        ros::shutdown();
    }
}

static bool createDirectoryRecursive(const std::string& path) {
    if (path.empty()) return false;
    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
        current += path[i];
        if (path[i] == '/' || i == path.size() - 1) {
            struct stat st;
            if (stat(current.c_str(), &st) != 0) {
                if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
    }
    return true;
}

void saveWorker(ThreadSafeSaveQueue& saveQueue,
                const PipelineConfig& config,
                SaveStats& stats) {
    std::cout << "[Frame Saver Worker] Started on thread: " << std::this_thread::get_id() << std::endl;
    const std::string baseDir = config.save_frames.directory;
    const std::string preDir = baseDir + "/pre";
    const std::string detDir = baseDir + "/detection";
    const std::string maskDir = baseDir + "/mask";

    createDirectoryRecursive(baseDir);
    if (config.save_frames.save_pre_inference) createDirectoryRecursive(preDir);
    if (config.save_frames.save_detections)    createDirectoryRecursive(detDir);
    if (config.save_frames.save_masks || config.save_frames.save_combined_mask) createDirectoryRecursive(maskDir);

    const std::string imgExt = config.save_frames.format_images.empty() ? "jpg" : config.save_frames.format_images;
    const std::vector<int> jpgParams = {cv::IMWRITE_JPEG_QUALITY, config.save_frames.jpeg_quality};
    const std::vector<int> pngParams = {cv::IMWRITE_PNG_COMPRESSION, 1}; // Fast lossless PNG

    SavedFrameBundle bundle;
    while (saveQueue.pop(bundle, g_running)) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int imagesSavedInBundle = 0;
        char filename[512];

        if (bundle.type == SavedFrameBundle::PRE_CAPTURE) {
            // 1. Pre-inference camera images directly captured from captureWorker
            if (config.save_frames.save_pre_inference) {
                for (size_t c = 0; c < bundle.rawBayerImages.size(); ++c) {
                    if (bundle.rawBayerImages[c].empty()) continue;
                    // Debayer from raw Bayer to BGR using the camera's bayer format
                    const int bayerCode = ((bundle.swapRBMask >> c) & 1u) ? cv::COLOR_BayerRG2BGR : cv::COLOR_BayerBG2BGR;
                    cv::Mat bgr;
                    cv::cvtColor(bundle.rawBayerImages[c], bgr, bayerCode);

                    std::string serial = (c < bundle.cameraSerials.size()) ? bundle.cameraSerials[c] : "cam";
                    std::snprintf(filename, sizeof(filename), "%s/frame_%06d_cam%d_%s_pre.%s",
                                  preDir.c_str(), bundle.frameId, (int)c, serial.c_str(), imgExt.c_str());
                    if (imgExt == "png") {
                        cv::imwrite(filename, bgr, pngParams);
                    } else {
                        cv::imwrite(filename, bgr, jpgParams);
                    }
                    imagesSavedInBundle++;
                }
            }
        } else if (bundle.type == SavedFrameBundle::INFERENCE_RESULT) {
            // 2. Detection images with overlays from inferenceWorker
            if (config.save_frames.save_detections) {
                for (size_t c = 0; c < bundle.detectionImages.size(); ++c) {
                    if (bundle.detectionImages[c].empty()) continue;
                    std::string serial = (c < bundle.cameraSerials.size()) ? bundle.cameraSerials[c] : "cam";
                    std::snprintf(filename, sizeof(filename), "%s/frame_%06d_cam%d_%s_det.%s",
                                  detDir.c_str(), bundle.frameId, (int)c, serial.c_str(), imgExt.c_str());
                    if (imgExt == "png") {
                        cv::imwrite(filename, bundle.detectionImages[c], pngParams);
                    } else {
                        cv::imwrite(filename, bundle.detectionImages[c], jpgParams);
                    }
                    imagesSavedInBundle++;
                }
            }

            // 3. Masks (Individual camera masks and horizontal combined mask)
            if (config.save_frames.save_masks) {
                for (size_t c = 0; c < bundle.maskImages.size(); ++c) {
                    if (bundle.maskImages[c].empty()) continue;
                    std::string serial = (c < bundle.cameraSerials.size()) ? bundle.cameraSerials[c] : "cam";
                    std::snprintf(filename, sizeof(filename), "%s/frame_%06d_cam%d_%s_mask.png",
                                  maskDir.c_str(), bundle.frameId, (int)c, serial.c_str());
                    cv::imwrite(filename, bundle.maskImages[c], pngParams);
                    imagesSavedInBundle++;
                }
            }

            if (config.save_frames.save_combined_mask && !bundle.combinedMask.empty()) {
                std::snprintf(filename, sizeof(filename), "%s/frame_%06d_combined_mask.png",
                              maskDir.c_str(), bundle.frameId);
                cv::imwrite(filename, bundle.combinedMask, pngParams);
                imagesSavedInBundle++;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double durMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats.totalBundlesSaved++;
        stats.totalImagesSaved += imagesSavedInBundle;
        stats.totalSaveTimeMs += durMs;

        if (stats.totalBundlesSaved % 50 == 0) {
            std::cout << "[FRAME SAVER] Saved " << stats.totalBundlesSaved << " frame bundles ("
                      << stats.totalImagesSaved << " files) to " << baseDir 
                      << " (queue size: " << saveQueue.size() << ")\n" << std::flush;
        }
    }

    std::cout << "[Frame Saver Worker] Stopped. Drained all items. Total saved: "
              << stats.totalBundlesSaved << " bundles (" << stats.totalImagesSaved << " files).\n";
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
                   CaptureStats& stats,
                   ThreadSafeSaveQueue* saveQueue = nullptr,
                   const PipelineConfig* config = nullptr,
                   uint32_t swapRBMask = 0) {
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

        // Push pre-inference frames directly from captureWorker so 100% of camera frames are saved
        if (saveQueue && config && config->save_frames.enabled && config->save_frames.save_pre_inference) {
            SavedFrameBundle preBundle;
            preBundle.type = SavedFrameBundle::PRE_CAPTURE;
            preBundle.frameId = bundle.frameId;
            preBundle.ptpTimeStampNs = bundle.ptpTimeStampNs;
            preBundle.cameraSerials = config->camera_serials;
            preBundle.swapRBMask = swapRBMask;
            preBundle.rawBayerImages.resize(numCams);
            for (int i = 0; i < numCams; i++) {
                preBundle.rawBayerImages[i] = bundle.images[i].clone();
            }
            saveQueue->push(std::move(preBundle));
        }

        auto t_cap_end = std::chrono::high_resolution_clock::now();
        double cap_duration_ms = std::chrono::duration<double, std::milli>(t_cap_end - t_cap_start).count();
        double cap_interval_ms = std::chrono::duration<double, std::milli>(t_cap_end - t_prev_capture).count();
        t_prev_capture = t_cap_end;
        double inst_cap_fps = (frameCounter > 1 && cap_interval_ms > 0.0) ? (1000.0 / cap_interval_ms) : (1000.0 / cap_duration_ms);
        static double smooth_cap_fps = 0.0;
        smooth_cap_fps = (smooth_cap_fps <= 0.0) ? inst_cap_fps : (0.85 * smooth_cap_fps + 0.15 * inst_cap_fps);

        stats.totalFrames++;
        stats.sumWorkMs += cap_duration_ms;
        stats.endTime = t_cap_end;

        g_latestCapFps.store(static_cast<float>(smooth_cap_fps));
        g_latestCapWorkMs.store(static_cast<float>(cap_duration_ms));

        // Push the synchronized bundle into the thread-safe queue
        queue.push(std::move(bundle));

        {
            std::lock_guard<std::mutex> lock(g_coutMutex);
            std::cout << "[CAPTURE]   Frame " << frameCounter
                      << " | Work: " << cap_duration_ms << " ms"
                      << " | Cycle: " << cap_interval_ms << " ms"
                      << " | Rate: " << smooth_cap_fps << " FPS"
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
                     uint32_t swapRBMask,
                     const PipelineConfig& config,
                     ros::Publisher* maskPub = nullptr,
                     ros::Publisher* maskVizPub = nullptr,
                     ros::Publisher* detPub = nullptr
#if ENABLE_WEB_SERVER
                     , MJPEGWebServer* webServer = nullptr
#endif
                     , ThreadSafeSaveQueue* saveQueue = nullptr
) {
    std::cout << "[Inference Worker] Started on thread: " << std::this_thread::get_id() << std::endl;
 
    int inferCount = 0;
    auto t_prev_infer = std::chrono::high_resolution_clock::now();
 
    // Network input size (must match the engine and the tensors allocated in main)
    constexpr int NET_W = 640, NET_H = 640;
    BayerLetterboxParams lp = makeBayerLetterboxParams(info.origWid, info.origHeig, NET_W, NET_H);
    lp.swapRBMask = swapRBMask;   // cameras that deliver BGGR (flipped 180 deg) get R/B swapped in the kernel

    // One tracker for the whole rig if enabled in config: independent track lists per camera,
    // IDs unique across cameras. If tracking is disabled, tracker is bypassed completely.
    std::unique_ptr<tracker2d::MultiCamByteTracker> tracker;
    std::vector<tracker2d::Detection> trackDets;   // scratch for the decoded detections
    if (config.tracking.enabled) {
        tracker2d::Tracker2DConfig trackerCfg;
        tracker = std::make_unique<tracker2d::MultiCamByteTracker>(batchSize, trackerCfg);
        trackDets.reserve(300);
    }

    // Overlay for one finished batch: tracked boxes with IDs (tracking on) or raw detections (off),
    // and paints each detected object into combinedMask according to the mono8 mask codec
    auto renderBatch = [&](const float* out, const FrameBundle& bundle, cv::Mat& combinedMask, int* outTargets = nullptr, std::vector<cv::Mat>* outPreImages = nullptr) {
        if (config.tracking.enabled && tracker) {
            return trackAndDraw(out, batchSize, bundle.images, info, swapRBMask, *tracker, trackDets, combinedMask, config.mask_options, 300, outTargets, outPreImages);
        } else {
            return drawDetections(out, batchSize, bundle.images, info, swapRBMask, combinedMask, config.mask_options, 300, config.confidence_threshold, outTargets, outPreImages);
        }
    };
 
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

    // Double-buffered CUDA events for pure GPU hardware timing (independent of CPU queue waits)
    cudaEvent_t trtStart[2];
    cudaEvent_t trtStop[2];
    for (int i = 0; i < 2; ++i) {
        cudaEventCreate(&trtStart[i]);
        cudaEventCreate(&trtStop[i]);
    }

    int currentSlot = 0;
    int nextSlot = 1;

    // --- PIPELINE SEEDING (Batch 0) ---
    FrameBundle currentBundle;
    if (!queue.pop(currentBundle, g_running)) {
        std::cout << "[Inference Worker] Exiting cleanly before starting...\n";
        cudaStreamDestroy(preProcess);
        cudaFreeHost(rawPinned);
        cudaFree(rawDev);
        for (int i = 0; i < 2; ++i) {
            cudaEventDestroy(trtStart[i]);
            cudaEventDestroy(trtStop[i]);
        }
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

    // Launch TensorRT on Batch 0 asynchronously with hardware event timing
    cudaEventRecord(trtStart[currentSlot], stream);
    context->setTensorAddress(inputName, inputs[currentSlot]);
    context->setTensorAddress(outputName, outputs[currentSlot]);
    context->enqueueV3(stream);
    cudaEventRecord(trtStop[currentSlot], stream);

    double smooth_infer_fps = 0.0;
    double smooth_trt_fps = 0.0;
    double smooth_compute_fps = 0.0;

    // --- STEADY-STATE DOUBLE-BUFFERED PIPELINE LOOP ---
    while (g_running) {
        FrameBundle nextBundle;
        // 1. Pop next bundle from queue (blocks on camera 20 Hz frame arrival)
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

        // Hardware-exact TensorRT GPU execution time (microsecond precision, zero CPU wait)
        float trt_ms = 0.0f;
        cudaEventElapsedTime(&trt_ms, trtStart[currentSlot], trtStop[currentSlot]);

        // 4. IMMEDIATELY launch TensorRT on the nextSlot (already preprocessed in background!)
        //    This keeps the GPU 100% saturated with ZERO idle wait time.
        inferCount++;
        cudaEventRecord(trtStart[nextSlot], stream);
        context->setTensorAddress(inputName, inputs[nextSlot]);
        context->setTensorAddress(outputName, outputs[nextSlot]);
        context->enqueueV3(stream);
        cudaEventRecord(trtStop[nextSlot], stream);

        // 5. Post-process (Tracking + Debayer + Drawing + Mask Generation) on currentSlot
        auto t_post_start = std::chrono::high_resolution_clock::now();
        cv::Mat combinedMask;
        int numTargets = 0;
        std::vector<cv::Mat> results = renderBatch(static_cast<float*>(outputs[currentSlot]), currentBundle, combinedMask, &numTargets);
        auto t_post_end = std::chrono::high_resolution_clock::now();
        double post_ms = std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();

        // 5a. Push detection and mask results to save queue if enabled
        if (config.save_frames.enabled && saveQueue &&
            (config.save_frames.save_detections || config.save_frames.save_masks || config.save_frames.save_combined_mask)) {
            SavedFrameBundle saveBundle;
            saveBundle.type = SavedFrameBundle::INFERENCE_RESULT;
            saveBundle.frameId = currentBundle.frameId;
            saveBundle.ptpTimeStampNs = currentBundle.ptpTimeStampNs;
            saveBundle.cameraSerials = config.camera_serials;

            if (config.save_frames.save_detections) {
                saveBundle.detectionImages.resize(results.size());
                for (size_t i = 0; i < results.size(); ++i) {
                    saveBundle.detectionImages[i] = results[i].clone();
                }
            }

            if (config.save_frames.save_masks) {
                saveBundle.maskImages.resize(batchSize);
                for (int b = 0; b < batchSize; ++b) {
                    int x = b * info.origWid;
                    if (x + info.origWid <= combinedMask.cols) {
                        saveBundle.maskImages[b] = combinedMask(cv::Rect(x, 0, info.origWid, info.origHeig)).clone();
                    }
                }
            }

            if (config.save_frames.save_combined_mask) {
                saveBundle.combinedMask = combinedMask.clone();
            }

            saveQueue->push(std::move(saveBundle));
        }

        // 5b. Publish horizontally combined mask and visualization topics to ROS One node
        if (config.ros.enabled && ros::ok()) {
            static std::unique_ptr<ros::NodeHandle> lazyNh;
            static std::unique_ptr<ros::Publisher> lazyPub;
            static std::unique_ptr<ros::Publisher> lazyVizPub;
            static std::unique_ptr<ros::Publisher> lazyDetPub;
            if (!maskPub) {
                static int retryCounter = 0;
                if (++retryCounter % 50 == 1 && ros::master::check()) {
                    lazyNh = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle());
                    lazyPub = std::unique_ptr<ros::Publisher>(new ros::Publisher(
                        lazyNh->advertise<sensor_msgs::Image>(config.ros.mask_topic, config.ros.queue_size)));
                    lazyVizPub = std::unique_ptr<ros::Publisher>(new ros::Publisher(
                        lazyNh->advertise<sensor_msgs::Image>(config.ros.mask_topic + "/viz", 2)));
                    lazyDetPub = std::unique_ptr<ros::Publisher>(new ros::Publisher(
                        lazyNh->advertise<sensor_msgs::Image>("/camera_array/detections", 2)));
                    maskPub = lazyPub.get();
                    maskVizPub = lazyVizPub.get();
                    detPub = lazyDetPub.get();
                    std::cout << "[ROS 1] Connected to ROS master! Now publishing on:\n" 
                              << "        - " << config.ros.mask_topic << " (raw mono8 mask)\n"
                              << "        - " << config.ros.mask_topic << "/viz (colorized RViz mask)\n"
                              << "        - /camera_array/detections (multi-camera overlay)" << std::endl;
                }
            }

            if (maskPub) {
                std_msgs::Header header;
                header.seq = currentBundle.frameId;
                if (currentBundle.ptpTimeStampNs > 0) {
                    header.stamp.fromNSec(currentBundle.ptpTimeStampNs);
                } else {
                    header.stamp = ros::Time::now();
                }
                header.frame_id = config.ros.frame_id;

                const std::string encoding = config.mask_options.single_channel 
                    ? sensor_msgs::image_encodings::MONO8 
                    : sensor_msgs::image_encodings::BGR8;

                sensor_msgs::ImagePtr msg(new sensor_msgs::Image());
                msg->header = header;
                msg->height = combinedMask.rows;
                msg->width = combinedMask.cols;
                msg->encoding = encoding;
                msg->is_bigendian = false;
                msg->step = static_cast<sensor_msgs::Image::_step_type>(combinedMask.step);
                const size_t dataSize = combinedMask.step * combinedMask.rows;
                msg->data.resize(dataSize);
                if (!combinedMask.empty()) {
                    std::memcpy(msg->data.data(), combinedMask.data, dataSize);
                }

                maskPub->publish(msg);

                // Publish colorized mask for RViz if someone is viewing it
                if (maskVizPub && maskVizPub->getNumSubscribers() > 0 && !combinedMask.empty()) {
                    cv::Mat viz;
                    if (combinedMask.channels() == 1) {
                        cv::Mat scaled;
                        combinedMask.convertTo(scaled, CV_8UC1, 7.0);
                        cv::applyColorMap(scaled, viz, cv::COLORMAP_JET);
                        viz.setTo(cv::Scalar(0, 0, 0), combinedMask == 0);
                    } else {
                        viz = combinedMask;
                    }

                    sensor_msgs::ImagePtr vizMsg(new sensor_msgs::Image());
                    vizMsg->header = header;
                    vizMsg->height = viz.rows;
                    vizMsg->width = viz.cols;
                    vizMsg->encoding = sensor_msgs::image_encodings::BGR8;
                    vizMsg->is_bigendian = false;
                    vizMsg->step = static_cast<sensor_msgs::Image::_step_type>(viz.step);
                    const size_t vizSize = viz.step * viz.rows;
                    vizMsg->data.resize(vizSize);
                    std::memcpy(vizMsg->data.data(), viz.data, vizSize);
                    maskVizPub->publish(vizMsg);
                }

                // Publish combined detection images for RViz if someone is viewing it
                if (detPub && detPub->getNumSubscribers() > 0 && !results.empty()) {
                    cv::Mat combinedDets;
                    cv::hconcat(results, combinedDets);

                    sensor_msgs::ImagePtr detMsg(new sensor_msgs::Image());
                    detMsg->header = header;
                    detMsg->height = combinedDets.rows;
                    detMsg->width = combinedDets.cols;
                    detMsg->encoding = sensor_msgs::image_encodings::BGR8;
                    detMsg->is_bigendian = false;
                    detMsg->step = static_cast<sensor_msgs::Image::_step_type>(combinedDets.step);
                    const size_t detSize = combinedDets.step * combinedDets.rows;
                    detMsg->data.resize(detSize);
                    std::memcpy(detMsg->data.data(), combinedDets.data, detSize);
                    detPub->publish(detMsg);
                }

                ros::spinOnce();
            }
        }

        // True active work latency on this frame (Pre + TRT GPU + Post/Tracking)
        double total_work_ms = last_pre_ms + trt_ms + post_ms;

        // 1. Raw TensorRT GPU throughput capability (before tracking / CPU post-processing)
        double inst_trt_fps = (trt_ms > 0.0f) ? (1000.0 / trt_ms) : 0.0;
        smooth_trt_fps = (smooth_trt_fps <= 0.0) ? inst_trt_fps : (0.85 * smooth_trt_fps + 0.15 * inst_trt_fps);

        // 2. Full pipeline active compute capability (AFTER tracking, drawing, debayering, and mask painting)
        //    This drops as numTargets increases due to ByteTrack Kalman + matching + mask painting!
        double inst_compute_fps = (total_work_ms > 0.0) ? (1000.0 / total_work_ms) : 0.0;
        smooth_compute_fps = (smooth_compute_fps <= 0.0) ? inst_compute_fps : (0.85 * smooth_compute_fps + 0.15 * inst_compute_fps);

        // 3. FPS drop caused by tracking and post-processing
        double fps_drop = smooth_trt_fps - smooth_compute_fps;

        // 4. Pipeline delivery rate (sync rate throttled by camera shutter, e.g. 20 Hz)
        auto t_work_end = t_post_end;
        double infer_interval_ms = std::chrono::duration<double, std::milli>(t_work_end - t_prev_infer).count();
        t_prev_infer = t_work_end;
        double inst_delivery_fps = (inferCount > 1 && infer_interval_ms > 0.0) ? (1000.0 / infer_interval_ms) : inst_compute_fps;
        smooth_infer_fps = (smooth_infer_fps <= 0.0) ? inst_delivery_fps : (0.85 * smooth_infer_fps + 0.15 * inst_delivery_fps);

        stats.totalFrames++;
        stats.sumPreMs += last_pre_ms;
        stats.sumTrtMs += trt_ms;
        stats.sumPostMs += post_ms;
        stats.sumTotalMs += total_work_ms;
        stats.sumTargets += numTargets;
        stats.endTime = t_work_end;

#if ENABLE_WEB_SERVER
        if (webServer) {
            WebStats wStats;
            wStats.frameId = currentBundle.frameId;
            wStats.capFps = g_latestCapFps.load();
            wStats.infFps = smooth_infer_fps;
            wStats.capWorkMs = g_latestCapWorkMs.load();
            wStats.trtMs = trt_ms;
            wStats.packetDrops = g_packetDrops.load();
            wStats.numCams = batchSize;
            wStats.trtFps = smooth_trt_fps;
            wStats.postTrackFps = smooth_compute_fps;
            wStats.postMs = post_ms;
            wStats.totalTargets = numTargets;
            webServer->updateFrames(currentBundle.images, results, wStats);
            webServer->updateMask(combinedMask, currentBundle.frameId);
        }
#endif

        {
            std::lock_guard<std::mutex> lock(g_coutMutex);
            std::cout << "[INFERENCE] Frame " << currentBundle.frameId
                      << " | Targets: " << numTargets
                      << " | Pre: " << std::fixed << std::setprecision(1) << last_pre_ms << " ms"
                      << " | TRT: " << trt_ms << " ms (" << std::fixed << std::setprecision(1) << smooth_trt_fps << " FPS)"
                      << " | Post: " << post_ms << " ms"
                      << " | Post-Track: " << smooth_compute_fps << " FPS"
                      << " (Drop: -" << std::fixed << std::setprecision(1) << (fps_drop > 0.0 ? fps_drop : 0.0) << " FPS)"
                      << " | Sync: " << smooth_infer_fps << " FPS"
                      << std::endl;
        }

        // 6. Ping-pong swap slots and hand over bundle
        last_pre_ms = pre_ms;
        currentBundle = std::move(nextBundle);
        std::swap(currentSlot, nextSlot);
    }

    // Drain the final launched batch
    cudaStreamSynchronize(stream);
    float final_trt_ms = 0.0f;
    cudaEventElapsedTime(&final_trt_ms, trtStart[currentSlot], trtStop[currentSlot]);

    auto t_drain_post_start = std::chrono::high_resolution_clock::now();
    cv::Mat finalMask;
    int finalTargets = 0;
    std::vector<cv::Mat> finalResults = renderBatch(static_cast<float*>(outputs[currentSlot]), currentBundle, finalMask, &finalTargets);
    auto t_drain_post_end = std::chrono::high_resolution_clock::now();
    double final_post_ms = std::chrono::duration<double, std::milli>(t_drain_post_end - t_drain_post_start).count();

    if (config.save_frames.enabled && saveQueue &&
        (config.save_frames.save_detections || config.save_frames.save_masks || config.save_frames.save_combined_mask)) {
        SavedFrameBundle saveBundle;
        saveBundle.type = SavedFrameBundle::INFERENCE_RESULT;
        saveBundle.frameId = currentBundle.frameId;
        saveBundle.ptpTimeStampNs = currentBundle.ptpTimeStampNs;
        saveBundle.cameraSerials = config.camera_serials;

        if (config.save_frames.save_detections) {
            saveBundle.detectionImages.resize(finalResults.size());
            for (size_t i = 0; i < finalResults.size(); ++i) {
                saveBundle.detectionImages[i] = finalResults[i].clone();
            }
        }

        if (config.save_frames.save_masks) {
            saveBundle.maskImages.resize(batchSize);
            for (int b = 0; b < batchSize; ++b) {
                int x = b * info.origWid;
                if (x + info.origWid <= finalMask.cols) {
                    saveBundle.maskImages[b] = finalMask(cv::Rect(x, 0, info.origWid, info.origHeig)).clone();
                }
            }
        }

        if (config.save_frames.save_combined_mask) {
            saveBundle.combinedMask = finalMask.clone();
        }

        saveQueue->push(std::move(saveBundle));
    }

    // Publish final horizontally combined mask and visualization topics to ROS One node
    if (maskPub && config.ros.enabled && ros::ok()) {
        std_msgs::Header header;
        header.seq = currentBundle.frameId;
        if (currentBundle.ptpTimeStampNs > 0) {
            header.stamp.fromNSec(currentBundle.ptpTimeStampNs);
        } else {
            header.stamp = ros::Time::now();
        }
        header.frame_id = config.ros.frame_id;

        const std::string encoding = config.mask_options.single_channel 
            ? sensor_msgs::image_encodings::MONO8 
            : sensor_msgs::image_encodings::BGR8;

        sensor_msgs::ImagePtr msg(new sensor_msgs::Image());
        msg->header = header;
        msg->height = finalMask.rows;
        msg->width = finalMask.cols;
        msg->encoding = encoding;
        msg->is_bigendian = false;
        msg->step = static_cast<sensor_msgs::Image::_step_type>(finalMask.step);
        const size_t dataSize = finalMask.step * finalMask.rows;
        msg->data.resize(dataSize);
        if (!finalMask.empty()) {
            std::memcpy(msg->data.data(), finalMask.data, dataSize);
        }

        maskPub->publish(msg);

        if (maskVizPub && maskVizPub->getNumSubscribers() > 0 && !finalMask.empty()) {
            cv::Mat viz;
            if (finalMask.channels() == 1) {
                cv::Mat scaled;
                finalMask.convertTo(scaled, CV_8UC1, 7.0);
                cv::applyColorMap(scaled, viz, cv::COLORMAP_JET);
                viz.setTo(cv::Scalar(0, 0, 0), finalMask == 0);
            } else {
                viz = finalMask;
            }

            sensor_msgs::ImagePtr vizMsg(new sensor_msgs::Image());
            vizMsg->header = header;
            vizMsg->height = viz.rows;
            vizMsg->width = viz.cols;
            vizMsg->encoding = sensor_msgs::image_encodings::BGR8;
            vizMsg->is_bigendian = false;
            vizMsg->step = static_cast<sensor_msgs::Image::_step_type>(viz.step);
            const size_t vizSize = viz.step * viz.rows;
            vizMsg->data.resize(vizSize);
            std::memcpy(vizMsg->data.data(), viz.data, vizSize);
            maskVizPub->publish(vizMsg);
        }

        if (detPub && detPub->getNumSubscribers() > 0 && !finalResults.empty()) {
            cv::Mat combinedDets;
            cv::hconcat(finalResults, combinedDets);

            sensor_msgs::ImagePtr detMsg(new sensor_msgs::Image());
            detMsg->header = header;
            detMsg->height = combinedDets.rows;
            detMsg->width = combinedDets.cols;
            detMsg->encoding = sensor_msgs::image_encodings::BGR8;
            detMsg->is_bigendian = false;
            detMsg->step = static_cast<sensor_msgs::Image::_step_type>(combinedDets.step);
            const size_t detSize = combinedDets.step * combinedDets.rows;
            detMsg->data.resize(detSize);
            std::memcpy(detMsg->data.data(), combinedDets.data, detSize);
            detPub->publish(detMsg);
        }

        ros::spinOnce();
    }

    double final_total_ms = last_pre_ms + final_trt_ms + final_post_ms;
    double final_trt_fps = (final_trt_ms > 0.0f) ? (1000.0 / final_trt_ms) : 0.0;
    double final_compute_fps = (final_total_ms > 0.0) ? (1000.0 / final_total_ms) : 0.0;
    double final_drop = final_trt_fps - final_compute_fps;

    auto t_drain_end = t_drain_post_end;
    double final_interval_ms = std::chrono::duration<double, std::milli>(t_drain_end - t_prev_infer).count();
    double final_delivery_fps = (inferCount > 1 && final_interval_ms > 0.0) ? (1000.0 / final_interval_ms) : final_compute_fps;

    stats.totalFrames++;
    stats.sumPreMs += last_pre_ms;
    stats.sumTrtMs += final_trt_ms;
    stats.sumPostMs += final_post_ms;
    stats.sumTotalMs += final_total_ms;
    stats.sumTargets += finalTargets;
    stats.endTime = t_drain_end;

#if ENABLE_WEB_SERVER
    if (webServer) {
        WebStats wStats;
        wStats.frameId = currentBundle.frameId;
        wStats.capFps = g_latestCapFps.load();
        wStats.infFps = final_delivery_fps;
        wStats.capWorkMs = g_latestCapWorkMs.load();
        wStats.trtMs = final_trt_ms;
        wStats.packetDrops = g_packetDrops.load();
        wStats.numCams = batchSize;
        wStats.trtFps = final_trt_fps;
        wStats.postTrackFps = final_compute_fps;
        wStats.postMs = final_post_ms;
        wStats.totalTargets = finalTargets;
        webServer->updateFrames(currentBundle.images, finalResults, wStats);
        webServer->updateMask(finalMask, currentBundle.frameId);
    }
#endif

    {
        std::lock_guard<std::mutex> lock(g_coutMutex);
        std::cout << "[INFERENCE] Frame " << currentBundle.frameId
                  << " | Targets: " << finalTargets
                  << " | Pre: " << std::fixed << std::setprecision(1) << last_pre_ms << " ms"
                  << " | TRT: " << final_trt_ms << " ms (" << std::fixed << std::setprecision(1) << final_trt_fps << " FPS)"
                  << " | Post: " << final_post_ms << " ms"
                  << " | Post-Track: " << final_compute_fps << " FPS"
                  << " (Drop: -" << std::fixed << std::setprecision(1) << (final_drop > 0.0 ? final_drop : 0.0) << " FPS)"
                  << " | Sync: " << final_delivery_fps << " FPS"
                  << std::endl;
    }

    std::cout << "[Inference Worker] Exiting cleanly...\n";
    cudaStreamDestroy(preProcess);
    cudaFreeHost(rawPinned);
    cudaFree(rawDev);
    for (int i = 0; i < 2; ++i) {
        cudaEventDestroy(trtStart[i]);
        cudaEventDestroy(trtStop[i]);
    }
}



int main(int argc, char** argv) {
    std::string configPath = "config.yaml";
    if (argc > 1 && argv[1][0] != '-') {
        configPath = argv[1];
    } else {
        std::ifstream checkConfig(configPath);
        if (!checkConfig.good()) {
            std::ifstream checkSub("config/config.yaml");
            if (checkSub.good()) {
                configPath = "config/config.yaml";
            }
        }
    }

    PipelineConfig config = PipelineConfig::loadFromFile(configPath);

    // Initialize ROS One (ROS 1)
    if (config.ros.enabled) {
        ros::init(argc, argv, config.ros.node_name, ros::init_options::NoSigintHandler);
    }

    std::unique_ptr<ros::NodeHandle> nh;
    std::unique_ptr<ros::Publisher> maskPub;
    std::unique_ptr<ros::Publisher> maskVizPub;
    std::unique_ptr<ros::Publisher> detPub;
    if (config.ros.enabled) {
        if (ros::master::check()) {
            nh = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle());
            ros::NodeHandle pnh("~");
            pnh.param("engine_file", config.engine_file, config.engine_file);
            pnh.param("mask_topic", config.ros.mask_topic, config.ros.mask_topic);
            pnh.param("frame_id", config.ros.frame_id, config.ros.frame_id);

            maskPub = std::unique_ptr<ros::Publisher>(new ros::Publisher(
                nh->advertise<sensor_msgs::Image>(config.ros.mask_topic, config.ros.queue_size)));
            maskVizPub = std::unique_ptr<ros::Publisher>(new ros::Publisher(
                nh->advertise<sensor_msgs::Image>(config.ros.mask_topic + "/viz", 2)));
            detPub = std::unique_ptr<ros::Publisher>(new ros::Publisher(
                nh->advertise<sensor_msgs::Image>("/camera_array/detections", 2)));

            std::cout << "[ROS 1] Node initialized ('" << config.ros.node_name 
                      << "'). Connected to master!\n"
                      << "        Mask Topic (raw mono8):       " << config.ros.mask_topic << "\n"
                      << "        Mask Topic (colorized RViz):  " << config.ros.mask_topic << "/viz\n"
                      << "        Detections Topic (RViz view): /camera_array/detections" << std::endl;
        } else {
            std::cout << "[ROS 1 INFO] ROS master (roscore) not detected at startup. "
                      << "Pipeline will run normally and connect to ROS once roscore is active.\n"
                      << "        Configured Mask Topic: " << config.ros.mask_topic << std::endl;
        }
    }

    config.printSummary();

    std::string engineFilePath = config.engine_file;

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
        std::vector<Arena::DeviceInfo> detectedCameras = arSystem->GetDevices();

        if (detectedCameras.empty()) {
            std::cout << "No LUCID camera found!!!!" << std::endl;
            Arena::CloseSystem(arSystem);
            return -1;
        }

        std::cout << "Found: " << detectedCameras.size() << " connected LUCID camera(s) on network:\n";
        for (size_t i = 0; i < detectedCameras.size(); ++i) {
            std::cout << "  - Serial: " << detectedCameras[i].SerialNumber().c_str() 
                      << " (" << detectedCameras[i].ModelName().c_str() << ")\n";
        }

        // Map connected cameras by serial number
        std::map<std::string, size_t> serialToIndex;
        for (size_t i = 0; i < detectedCameras.size(); ++i) {
            serialToIndex[detectedCameras[i].SerialNumber().c_str()] = i;
        }

        // Order cameras according to config.camera_serials
        std::vector<Arena::DeviceInfo> orderedCameras;
        if (!config.camera_serials.empty()) {
            for (const auto& serial : config.camera_serials) {
                auto it = serialToIndex.find(serial);
                if (it != serialToIndex.end()) {
                    orderedCameras.push_back(detectedCameras[it->second]);
                    serialToIndex.erase(it);
                } else {
                    std::cerr << "[CONFIG WARN] Camera serial '" << serial 
                              << "' from config is NOT connected!\n";
                }
            }
        }

        if (orderedCameras.empty()) {
            std::cout << "[CONFIG WARN] No configured serials matched! Using all detected cameras in enumeration order.\n";
            orderedCameras = detectedCameras;
        } else if (!serialToIndex.empty()) {
            std::cout << "[CONFIG NOTE] " << serialToIndex.size() 
                      << " connected camera(s) not listed in config will not be used in this run.\n";
        }

        int numCams = orderedCameras.size();
        std::cout << "[PIPELINE] Active camera order for horizontal mask (" << numCams << " cameras):\n";
        for (int i = 0; i < numCams; ++i) {
            std::cout << "  [" << i << "] Serial: " << orderedCameras[i].SerialNumber().c_str()
                      << " | Model: " << orderedCameras[i].ModelName().c_str()
                      << " | Mask H-Slice: [" << i * 1920 << " .. " << (i + 1) * 1920 << " px)\n";
        }

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
            arDevice[i] = arSystem->CreateDevice(orderedCameras[i]);
            arNodeMap[i] = arDevice[i]->GetNodeMap();
            auto* arStreamNodeMap = arDevice[i]->GetTLStreamNodeMap();

            //configuring each camera

            //pixel formatting
            const std::string serial = orderedCameras[i].SerialNumber().c_str();
            const bool rotate180 = config.rotate_180_serials.count(serial) > 0;

            // 1. Clear any flip left over from a previous run
            Arena::SetNodeValue<bool>(arNodeMap[i], "ReverseX", false);
            Arena::SetNodeValue<bool>(arNodeMap[i], "ReverseY", false);

            // 2. With no flip active, BayerRG8 is writable
            Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "PixelFormat", "BayerRG8");

            // 3. Apply the flip; the camera updates PixelFormat itself
            Arena::SetNodeValue<bool>(arNodeMap[i], "ReverseX", rotate180);
            Arena::SetNodeValue<bool>(arNodeMap[i], "ReverseY", rotate180);

            const CameraSetting camCfg = config.getCameraSetting(serial);

            GenICam::gcstring pf = Arena::GetNodeValue<GenICam::gcstring>(arNodeMap[i], "PixelFormat");
            std::cout << "  Camera [" << i << "] " << serial
                      << " | rotate180=" << rotate180
                      << " | PixelFormat=" << pf
                      << " | ExpAuto=" << camCfg.exposure_auto
                      << " | ExpTime=" << camCfg.exposure_time << " us"
                      << " | GainAuto=" << camCfg.gain_auto
                      << " | Gain=" << camCfg.gain << " dB\n" << std::flush;

            if (pf == "BayerBG8") {
                swapRBMask |= (1u << i);
            } else if (pf != "BayerRG8") {
                std::cerr << "[WARN] Camera " << i << " reports " << pf
                          << " - kernel only handles BayerRG8 / BayerBG8\n";
            }

            // Exposure & Gain configuration from config.yaml
            try {
                Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "ExposureAuto", camCfg.exposure_auto.c_str());
            } catch (const std::exception& e) {
                std::cerr << "[WARN] Camera " << serial << " could not set ExposureAuto=" << camCfg.exposure_auto << ": " << e.what() << "\n";
            }

            if (camCfg.exposure_auto == "Off") {
                try {
                    Arena::SetNodeValue<double>(arNodeMap[i], "ExposureTime", camCfg.exposure_time);
                } catch (const std::exception& e) {
                    std::cerr << "[WARN] Camera " << serial << " could not set ExposureTime=" << camCfg.exposure_time << ": " << e.what() << "\n";
                }
            }

            try {
                Arena::SetNodeValue<GenICam::gcstring>(arNodeMap[i], "GainAuto", camCfg.gain_auto.c_str());
            } catch (const std::exception& e) {
                std::cerr << "[WARN] Camera " << serial << " could not set GainAuto=" << camCfg.gain_auto << ": " << e.what() << "\n";
            }

            if (camCfg.gain_auto == "Off") {
                try {
                    Arena::SetNodeValue<double>(arNodeMap[i], "Gain", camCfg.gain);
                } catch (const std::exception& e) {
                    std::cerr << "[WARN] Camera " << serial << " could not set Gain=" << camCfg.gain << ": " << e.what() << "\n";
                }
            }

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
        MJPEGWebServer webServer(config.web_server.port);
        std::vector<CameraDetail> camDetails(numCams);
        for (int i = 0; i < numCams; ++i) {
            camDetails[i].index = i;
            camDetails[i].serial = orderedCameras[i].SerialNumber().c_str();
            camDetails[i].model = orderedCameras[i].ModelName().c_str();
            camDetails[i].ip = orderedCameras[i].IpAddressStr().c_str();
        }
        webServer.setCameraDetails(camDetails);

        if (config.web_server.enabled) {
            if (!webServer.start()) {
                std::cerr << "[WARN] Failed to start Web Server on port " << config.web_server.port << "\n";
            }
        }
#endif

        std::cout << "\n[PIPELINE] Launching Trigger (20Hz), Capture, and Inference threads. Press Ctrl+C to stop...\n\n";

        // Optional Frame Saver thread & queue
        std::unique_ptr<ThreadSafeSaveQueue> saveQueue;
        std::unique_ptr<std::thread> saveThread;
        SaveStats saveStats;
        if (config.save_frames.enabled) {
            saveQueue = std::make_unique<ThreadSafeSaveQueue>(config.save_frames.max_queue_size);
            g_saveQueuePtr = saveQueue.get();
            saveThread = std::make_unique<std::thread>(saveWorker, 
                                                       std::ref(*saveQueue), 
                                                       std::cref(config), 
                                                       std::ref(saveStats));
        }

        // Dedicated high-precision 20.0 Hz PTP Action Command generator
        std::thread triggerThread(triggerWorker, 
                                 arSystemTL, 
                                 arNodeMap[0]);

        // Capture worker ingesting frames as they arrive from all 4 cameras
        std::thread cameraThread(captureWorker, 
                                 numCams, 
                                 std::cref(arDevice), 
                                 std::ref(frameQueue),
                                 std::ref(capStats),
                                 saveQueue.get(),
                                 &config,
                                 swapRBMask);

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
                                  swapRBMask,
                                  config,
                                  maskPub.get(),
                                  maskVizPub.get(),
                                  detPub.get()
#if ENABLE_WEB_SERVER
                                  , config.web_server.enabled ? &webServer : nullptr
#endif
                                  , saveQueue.get()
        );

        cameraThread.join();
        triggerThread.join();
        inferenceThread.join();
        g_queuePtr = nullptr;

        if (saveQueue) {
            saveQueue->shutdown();
        }
        if (saveThread && saveThread->joinable()) {
            std::cout << "[INFO] Waiting for Frame Saver thread to drain and finish...\n" << std::flush;
            saveThread->join();
        }
        g_saveQueuePtr = nullptr;

#if ENABLE_WEB_SERVER
        if (config.web_server.enabled) {
            webServer.stop();
        }
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
            double avgPostMs = infStats.sumPostMs / infStats.totalFrames;
            double avgTotalMs = infStats.sumTotalMs / infStats.totalFrames;
            double avgTargets = static_cast<double>(infStats.sumTargets) / infStats.totalFrames;
            double rawTrtFps = avgTrtMs > 0.0 ? (1000.0 / avgTrtMs) : 0.0;
            double postTrackFps = avgTotalMs > 0.0 ? (1000.0 / avgTotalMs) : 0.0;
            double avgDropFps = rawTrtFps - postTrackFps;

            const bool trackingOn = config.tracking.enabled;
            std::cout << "[INFERENCE SUMMARY]:\n"
                      << "  Total Batches Inferred   : " << infStats.totalFrames << "\n"
                      << "  Total Images Processed   : " << infStats.totalFrames * batchSize << " (Batch Size: " << batchSize << ")\n"
                      << "  Tracking Mode            : " << (trackingOn ? "ENABLED (ByteTrack MultiCam)" : "DISABLED (Inference Only, Raw Detections in Mask)") << "\n"
                      << "  Avg " << (trackingOn ? "Tracked" : "Detected") << " Targets/Frame: " << std::fixed << std::setprecision(1) << avgTargets << " targets\n"
                      << "  Avg Preprocessing Time   : " << avgPreMs << " ms\n"
                      << "  Avg TensorRT GPU Time    : " << avgTrtMs << " ms (hardware execution via cudaEvent)\n"
                      << "  Avg Post/" << (trackingOn ? "Tracking" : "Processing") << " Time   : " << avgPostMs << " ms\n"
                      << "  Avg Active Frame Work    : " << avgTotalMs << " ms\n"
                      << "  Raw Engine Throughput    : " << rawTrtFps << " FPS (" << rawTrtFps * batchSize << " images/sec)\n"
                      << "  Post-" << (trackingOn ? "Track" : "Process") << " Compute FPS   : " << postTrackFps << " FPS (" << postTrackFps * batchSize << " images/sec)\n"
                      << "  " << (trackingOn ? "Tracking & Post" : "Post-Processing") << " FPS Drop : -" << (avgDropFps > 0 ? avgDropFps : 0.0) << " FPS ("
                      << (rawTrtFps > 0 ? (avgDropFps / rawTrtFps * 100.0) : 0.0) << " %)\n"
                      << "  Pipeline Delivery Rate   : " << avgInfRate << " FPS (camera synchronized)\n"
                      << "  Pipeline Ingestion Rate  : " << avgInfRate * batchSize << " images/sec\n\n";
        } else {
            std::cout << "[INFERENCE SUMMARY]: Not enough frames inferred to compute average.\n\n";
        }

        if (config.save_frames.enabled) {
            double avgSaveMs = saveStats.totalBundlesSaved > 0 
                ? (saveStats.totalSaveTimeMs / saveStats.totalBundlesSaved) 
                : 0.0;
            std::cout << "[FRAME SAVER SUMMARY]:\n"
                      << "  Total Frame Bundles Saved: " << saveStats.totalBundlesSaved << "\n"
                      << "  Total Image Files Written: " << saveStats.totalImagesSaved << "\n"
                      << "  Avg Bundle Write Time    : " << std::fixed << std::setprecision(2) << avgSaveMs << " ms\n"
                      << "  Dropped Queue Items      : " << (saveQueue ? saveQueue->getDroppedCount() : 0) << "\n"
                      << "  Target Directory         : " << config.save_frames.directory << "\n";
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