#include "histogram_kernel.h"

__global__ void histogramKernel(
    const unsigned char* images,
    int width, int height,
    unsigned int* histograms,
    int batchCount
) {
    extern __shared__ unsigned int sharedHist[];
    int imgIdx = blockIdx.x;
    if (imgIdx >= batchCount) return;

    unsigned char const* img = images + imgIdx * width * height;
    unsigned int* hist = histograms + imgIdx * 256;

    // zero shared histogram
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        sharedHist[i] = 0;
    __syncthreads();

    // each thread processes multiple pixels
    int total = width * height;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int idx = tid; idx < total; idx += stride) {
        unsigned char pix = img[idx];
        atomicAdd(&sharedHist[pix], 1);
    }
    __syncthreads();

    // write back to global memory
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        hist[i] = sharedHist[i];
}