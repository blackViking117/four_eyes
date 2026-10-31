#include "bayer_letterbox.h"

// Fused replacement for:
//   cv::cvtColor(bayer, bgr, cv::COLOR_BayerBG2BGR);        // = GenICam BayerRG8 (RGGB)
//   cv::resize(bgr, resized, {newW, newH});                 // INTER_LINEAR
//   letterbox into a 114-grey canvas, BGR->RGB, /255, HWC->CHW
// and, optionally, writes the debayered + resized BGR image to `preview` for visualisation.
//
// One thread per output pixel of the canvas. Only the demosaic taps the resize
// actually needs are computed (for 1920x1200 -> 640x400 that is one tap per pixel,
// because the 3x downscale lands exactly on integer source pixels).

// Bilinear demosaic of an RGGB mosaic at one pixel, same rounding as OpenCV.
// Assumes (0,0) is a red site (OffsetX / OffsetY even).
__device__ __forceinline__
void demosaicRGGB(const uint8_t* __restrict__ img, int W, int H, int x, int y,
                  int& r, int& g, int& b)
{
    // OpenCV fills the 1-pixel border with a copy of its inner neighbour,
    // which is the same as clamping the coordinate.
    x = max(min(x, W - 2), 1);
    y = max(min(y, H - 2), 1);

    const uint8_t* p = img + y * W + x;
    const int c  = p[0];
    const int l  = p[-1], rt = p[1];
    const int u  = p[-W], d  = p[W];
    const bool xo = (x & 1) != 0, yo = (y & 1) != 0;

    if (!yo && !xo) {            // R site
        r = c;
        g = (l + rt + u + d + 2) >> 2;
        b = (p[-W - 1] + p[-W + 1] + p[W - 1] + p[W + 1] + 2) >> 2;
    } else if (!yo && xo) {      // G site on an R row (R left/right, B above/below)
        g = c;
        r = (l + rt + 1) >> 1;
        b = (u + d + 1) >> 1;
    } else if (yo && !xo) {      // G site on a B row (B left/right, R above/below)
        g = c;
        b = (l + rt + 1) >> 1;
        r = (u + d + 1) >> 1;
    } else {                     // B site
        b = c;
        g = (l + rt + u + d + 2) >> 2;
        r = (p[-W - 1] + p[-W + 1] + p[W - 1] + p[W + 1] + 2) >> 2;
    }
}

__global__ void bayerLetterboxKernel(const uint8_t* __restrict__ raw,
                                     float* __restrict__ out,
                                     uint8_t* __restrict__ preview,      // may be nullptr
                                     BayerLetterboxParams p)
{
    const int dx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy  = blockIdx.y * blockDim.y + threadIdx.y;
    const int cam = blockIdx.z;
    if (dx >= p.dstW || dy >= p.dstH) return;

    int R = 114, G = 114, B = 114;                 // letterbox colour
    const int ix = dx - p.padX;
    const int iy = dy - p.padY;

    if (ix >= 0 && ix < p.newW && iy >= 0 && iy < p.newH) {
        const uint8_t* img = raw + static_cast<size_t>(cam) * p.srcW * p.srcH;

        // cv::resize INTER_LINEAR source mapping (pixel centres aligned)
        float fx = (ix + 0.5f) * (static_cast<float>(p.srcW) / p.newW) - 0.5f;
        float fy = (iy + 0.5f) * (static_cast<float>(p.srcH) / p.newH) - 0.5f;
        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        fx -= x0;
        fy -= y0;
        if (x0 < 0)            { x0 = 0;            fx = 0.f; }
        if (x0 >= p.srcW - 1)  { x0 = p.srcW - 1;   fx = 0.f; }
        if (y0 < 0)            { y0 = 0;            fy = 0.f; }
        if (y0 >= p.srcH - 1)  { y0 = p.srcH - 1;   fy = 0.f; }
        const int x1 = min(x0 + 1, p.srcW - 1);
        const int y1 = min(y0 + 1, p.srcH - 1);

        float accR = 0.f, accG = 0.f, accB = 0.f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float w = ((t & 1) ? fx : 1.f - fx) * ((t & 2) ? fy : 1.f - fy);
            if (w == 0.f) continue;                // exact-integer scales: only 1 tap survives
            int r, g, b;
            demosaicRGGB(img, p.srcW, p.srcH, (t & 1) ? x1 : x0, (t & 2) ? y1 : y0, r, g, b);
            accR += w * r;  accG += w * g;  accB += w * b;
        }

        // Cameras flipped 180 deg deliver BGGR. Bilinear demosaicing is symmetric in R and B, so
        // demosaicing as RGGB and swapping the two channels here gives the exact BGGR result.
        if ((p.swapRBMask >> cam) & 1u) {
            const float tmp = accR;
            accR = accB;
            accB = tmp;
        }

        // resize output is uint8 in the CPU path, so quantise here too
        R = min(static_cast<int>(accR + 0.5f), 255);
        G = min(static_cast<int>(accG + 0.5f), 255);
        B = min(static_cast<int>(accB + 0.5f), 255);

        // Optional preview: BGR, HWC, no padding. These are the values cv::resize would have
        // produced, so the CPU debayer is not needed again for display.
        if (preview != nullptr) {
            uint8_t* q = preview + ((static_cast<size_t>(cam) * p.newH + iy) * p.newW + ix) * 3;
            q[0] = static_cast<uint8_t>(B);
            q[1] = static_cast<uint8_t>(G);
            q[2] = static_cast<uint8_t>(R);
        }
    }

    // planar RGB (channel 0 = R), same layout imagePreprocessor_v4 writes
    const size_t plane = static_cast<size_t>(p.dstW) * p.dstH;
    float* o = out + static_cast<size_t>(cam) * 3 * plane + static_cast<size_t>(dy) * p.dstW + dx;
    o[0]         = R / 255.0f;
    o[plane]     = G / 255.0f;
    o[2 * plane] = B / 255.0f;
}

// ---- launcher ----
void launchBayerLetterbox(const uint8_t* raw, float* out, int batch,
                          const BayerLetterboxParams& p, cudaStream_t stream,
                          uint8_t* preview)
{
    const dim3 block(32, 8);
    const dim3 grid((p.dstW + block.x - 1) / block.x,
                    (p.dstH + block.y - 1) / block.y,
                    batch);
    bayerLetterboxKernel<<<grid, block, 0, stream>>>(raw, out, preview, p);
}