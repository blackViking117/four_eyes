#include <iostream>
#include <cstdint>

#define COARSE_FACTOR 16
#define IMG_DIM 640
#define TILE_WIDTH 16384

__global__ void imagePreprocessor_v4(const uint8_t* img, float* data) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int globalPixPart = bx * (TILE_WIDTH);

    // performing actual calculation and writing to global memory
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int localPix = tx + (blockDim.x * i);
        int globalPix = globalPixPart + localPix;

        data[globalPix + (IMG_DIM * IMG_DIM * 2)] = img[globalPix * 3] / 255.0f;
        data[globalPix + (IMG_DIM * IMG_DIM)]     = img[(globalPix * 3) + 1] / 255.0f;
        data[globalPix]                           = img[(globalPix * 3) + 2] / 255.0f;
    }
}