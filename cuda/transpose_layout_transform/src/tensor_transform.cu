#include "cuda/transpose_layout_transform.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;

int launch_blocks(const std::int64_t numel) {
    return static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
}

__global__ void concat_nchw_axis1_kernel(
    const float* a,
    const float* b,
    float* output,
    const int c_a,
    const int c_b,
    const int h,
    const int w,
    const int total) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int c_out = c_a + c_b;
    int tmp = linear;
    const int ow = tmp % w;
    tmp /= w;
    const int oh = tmp % h;
    tmp /= h;
    const int channel = tmp % c_out;
    const int batch = tmp / c_out;

    if (channel < c_a) {
        output[linear] = a[((batch * c_a + channel) * h + oh) * w + ow];
    } else {
        const int b_channel = channel - c_a;
        output[linear] = b[((batch * c_b + b_channel) * h + oh) * w + ow];
    }
}

__global__ void nchw_to_nhwc_kernel(const float* input, float* output, const int c, const int h, const int w, const int total) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    int tmp = linear;
    const int channel = tmp % c;
    tmp /= c;
    const int ow = tmp % w;
    tmp /= w;
    const int oh = tmp % h;
    const int batch = tmp / h;

    output[linear] = input[((batch * c + channel) * h + oh) * w + ow];
}

__global__ void nhwc_to_nchw_kernel(const float* input, float* output, const int c, const int h, const int w, const int total) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    int tmp = linear;
    const int ow = tmp % w;
    tmp /= w;
    const int oh = tmp % h;
    tmp /= h;
    const int channel = tmp % c;
    const int batch = tmp / c;

    output[linear] = input[((batch * h + oh) * w + ow) * c + channel];
}

__global__ void copy_1d_kernel(const float* input, float* output, const std::int64_t numel) {
    const std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        output[index] = input[index];
    }
}

}  // namespace

cudaError_t concat_nchw_axis1(
    const float* a,
    const float* b,
    float* output,
    const int n,
    const int c_a,
    const int c_b,
    const int h,
    const int w,
    cudaStream_t stream) {
    if (a == nullptr || b == nullptr || output == nullptr || n <= 0 || c_a <= 0 || c_b <= 0 || h <= 0 || w <= 0) {
        return cudaErrorInvalidValue;
    }
    const int total = n * (c_a + c_b) * h * w;
    concat_nchw_axis1_kernel<<<launch_blocks(total), kBlockSize, 0, stream>>>(a, b, output, c_a, c_b, h, w, total);
    return cudaGetLastError();
}

cudaError_t permute_nchw_to_nhwc(
    const float* input,
    float* output,
    const int n,
    const int c,
    const int h,
    const int w,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        return cudaErrorInvalidValue;
    }
    const int total = n * c * h * w;
    nchw_to_nhwc_kernel<<<launch_blocks(total), kBlockSize, 0, stream>>>(input, output, c, h, w, total);
    return cudaGetLastError();
}

cudaError_t permute_nhwc_to_nchw(
    const float* input,
    float* output,
    const int n,
    const int c,
    const int h,
    const int w,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        return cudaErrorInvalidValue;
    }
    const int total = n * c * h * w;
    nhwc_to_nchw_kernel<<<launch_blocks(total), kBlockSize, 0, stream>>>(input, output, c, h, w, total);
    return cudaGetLastError();
}

cudaError_t copy_1d(const float* input, float* output, const std::int64_t numel, cudaStream_t stream) {
    if (input == nullptr || output == nullptr || numel <= 0) {
        return cudaErrorInvalidValue;
    }
    copy_1d_kernel<<<launch_blocks(numel), kBlockSize, 0, stream>>>(input, output, numel);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
