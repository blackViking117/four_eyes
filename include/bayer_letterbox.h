#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// Everything the fused kernel needs to know about the geometry.
struct BayerLetterboxParams {
    int srcW, srcH;   // raw Bayer frame           (1920 x 1200)
    int dstW, dstH;   // network input canvas      (640 x 640, or 640 x 416 for a rect engine)
    int newW, newH;   // resized image inside it   (640 x 400)
    int padX, padY;   // its top-left in the canvas (0, 120)
};

// Same arithmetic as your CPU path (scale = min ratio, round, centre the image),
// generalised to a non-square canvas.
inline BayerLetterboxParams makeBayerLetterboxParams(int srcW, int srcH, int dstW, int dstH)
{
    const float scale = std::min(static_cast<float>(dstW) / srcW,
                                 static_cast<float>(dstH) / srcH);
    BayerLetterboxParams p;
    p.srcW = srcW;  p.srcH = srcH;
    p.dstW = dstW;  p.dstH = dstH;
    p.newW = static_cast<int>(std::round(srcW * scale));
    p.newH = static_cast<int>(std::round(srcH * scale));
    p.padX = (dstW - p.newW) / 2;
    p.padY = (dstH - p.newH) / 2;
    return p;
}

// Bytes needed for the optional BGR preview of a whole batch.
inline size_t bayerPreviewBytes(const BayerLetterboxParams& p, int batch)
{
    return static_cast<size_t>(batch) * p.newW * p.newH * 3;
}

// raw     : [batch][srcH][srcW] uint8 RGGB Bayer, must be device-accessible (cudaMalloc'd is best)
// out     : [batch][3][dstH][dstW] float, planar RGB, values in [0,1]   (this is the TensorRT input)
// preview : optional (nullptr = off). [batch][newH][newW][3] uint8 BGR, the exact debayered +
//           resized image the network sees WITHOUT the letterbox padding (640x400 for 1920x1200).
//           Wrap one camera with cv::Mat(p.newH, p.newW, CV_8UC3, preview + cam * p.newW * p.newH * 3).
//           Must be CPU-readable (cudaHostAllocMapped) if you want to draw on it from the host.
void launchBayerLetterbox(const uint8_t* raw, float* out, int batch,
                          const BayerLetterboxParams& p, cudaStream_t stream,
                          uint8_t* preview = nullptr);