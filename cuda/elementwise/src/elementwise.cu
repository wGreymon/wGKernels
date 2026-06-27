#include "cuda/elementwise.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;

int launch_blocks(const std::int64_t numel) {
    return static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
}

template <typename Op>
__global__ void binary_kernel(
    const float* a,
    const float* b,
    float* output,
    const std::int64_t numel,
    Op op) {
    const std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        output[index] = op(a[index], b[index]);
    }
}

struct AddOp {
    __device__ float operator()(const float a, const float b) const { return a + b; }
};

struct SubOp {
    __device__ float operator()(const float a, const float b) const { return a - b; }
};

struct MulOp {
    __device__ float operator()(const float a, const float b) const { return a * b; }
};

bool invalid_binary_args(const float* a, const float* b, const float* output, const std::int64_t numel) {
    return a == nullptr || b == nullptr || output == nullptr || numel <= 0;
}

}  // namespace

cudaError_t add(const float* a, const float* b, float* output, const std::int64_t numel, cudaStream_t stream) {
    if (invalid_binary_args(a, b, output, numel)) {
        return cudaErrorInvalidValue;
    }
    binary_kernel<<<launch_blocks(numel), kBlockSize, 0, stream>>>(a, b, output, numel, AddOp {});
    return cudaGetLastError();
}

cudaError_t sub(const float* a, const float* b, float* output, const std::int64_t numel, cudaStream_t stream) {
    if (invalid_binary_args(a, b, output, numel)) {
        return cudaErrorInvalidValue;
    }
    binary_kernel<<<launch_blocks(numel), kBlockSize, 0, stream>>>(a, b, output, numel, SubOp {});
    return cudaGetLastError();
}

cudaError_t mul(const float* a, const float* b, float* output, const std::int64_t numel, cudaStream_t stream) {
    if (invalid_binary_args(a, b, output, numel)) {
        return cudaErrorInvalidValue;
    }
    binary_kernel<<<launch_blocks(numel), kBlockSize, 0, stream>>>(a, b, output, numel, MulOp {});
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
