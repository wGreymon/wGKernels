#include "cuda/norm.hpp"

#include <cuda_runtime.h>

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;

__global__ void batchnorm2d_inference_nchw_kernel(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    const int c,
    const int h,
    const int w,
    const int total) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    const int spatial = h * w;
    const int channel = (linear / spatial) % c;
    output[linear] = input[linear] * scale[channel] + bias[channel];
}

}  // namespace

cudaError_t batchnorm2d_inference_nchw(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    const int n,
    const int c,
    const int h,
    const int w,
    cudaStream_t stream) {
    if (input == nullptr || scale == nullptr || bias == nullptr || output == nullptr || n <= 0 || c <= 0 ||
        h <= 0 || w <= 0) {
        return cudaErrorInvalidValue;
    }
    const int total = n * c * h * w;
    const int blocks = (total + kBlockSize - 1) / kBlockSize;
    batchnorm2d_inference_nchw_kernel<<<blocks, kBlockSize, 0, stream>>>(input, scale, bias, output, c, h, w, total);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
