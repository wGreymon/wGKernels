#include "cuda/activation.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;

int launch_blocks(const std::int64_t numel) {
    return static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
}

__global__ void silu_kernel(const float* input, float* output, const std::int64_t numel) {
    const std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        const float value = input[index];
        output[index] = value / (1.0f + expf(-value));
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, const std::int64_t numel) {
    const std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        output[index] = 1.0f / (1.0f + expf(-input[index]));
    }
}

__global__ void exp_kernel(const float* input, float* output, const std::int64_t numel) {
    const std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        output[index] = expf(input[index]);
    }
}

bool invalid_unary_args(const float* input, const float* output, const std::int64_t numel) {
    return input == nullptr || output == nullptr || numel <= 0;
}

}  // namespace

cudaError_t silu(const float* input, float* output, const std::int64_t numel, cudaStream_t stream) {
    if (invalid_unary_args(input, output, numel)) {
        return cudaErrorInvalidValue;
    }
    silu_kernel<<<launch_blocks(numel), kBlockSize, 0, stream>>>(input, output, numel);
    return cudaGetLastError();
}

cudaError_t sigmoid(const float* input, float* output, const std::int64_t numel, cudaStream_t stream) {
    if (invalid_unary_args(input, output, numel)) {
        return cudaErrorInvalidValue;
    }
    sigmoid_kernel<<<launch_blocks(numel), kBlockSize, 0, stream>>>(input, output, numel);
    return cudaGetLastError();
}

cudaError_t exp(const float* input, float* output, const std::int64_t numel, cudaStream_t stream) {
    if (invalid_unary_args(input, output, numel)) {
        return cudaErrorInvalidValue;
    }
    exp_kernel<<<launch_blocks(numel), kBlockSize, 0, stream>>>(input, output, numel);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
