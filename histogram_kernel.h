#ifndef HISTOGRAM_KERNEL_H
#define HISTOGRAM_KERNEL_H

#include <cuda_runtime.h>

// Compute per-image histograms: images is batch×W×H bytes, histograms is batch×256 ints
__global__ void histogramKernel(
    const unsigned char* __restrict__ images,
    int width, int height,
    unsigned int* __restrict__ histograms,
    int batchCount
);

#endif // HISTOGRAM_KERNEL_H