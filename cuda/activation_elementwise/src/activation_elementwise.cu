#include "wgkernel/cuda/activation_elementwise.hpp"

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

bool invalid_unary_args(const float* input, const float* output, const std::int64_t numel) {
    return input == nullptr || output == nullptr || numel <= 0;
}

bool invalid_binary_args(const float* a, const float* b, const float* output, const std::int64_t numel) {
    return a == nullptr || b == nullptr || output == nullptr || numel <= 0;
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
