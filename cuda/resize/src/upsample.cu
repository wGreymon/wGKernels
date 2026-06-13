#include "cuda/resize.hpp"

#include <cuda_runtime.h>

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;

__global__ void upsample_nearest2d_nchw_kernel(
    const float* input,
    float* output,
    const int c,
    const int h_in,
    const int w_in,
    const int h_out,
    const int w_out,
    const int total) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    int tmp = linear;
    const int ow = tmp % w_out;
    tmp /= w_out;
    const int oh = tmp % h_out;
    tmp /= h_out;
    const int channel = tmp % c;
    const int batch = tmp / c;

    const int ih = min(static_cast<int>(static_cast<long long>(oh) * h_in / h_out), h_in - 1);
    const int iw = min(static_cast<int>(static_cast<long long>(ow) * w_in / w_out), w_in - 1);
    output[linear] = input[((batch * c + channel) * h_in + ih) * w_in + iw];
}

__global__ void upsample_bilinear2d_nchw_kernel(
    const float* input,
    float* output,
    const int c,
    const int h_in,
    const int w_in,
    const int h_out,
    const int w_out,
    const bool align_corners,
    const int total) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    int tmp = linear;
    const int ow = tmp % w_out;
    tmp /= w_out;
    const int oh = tmp % h_out;
    tmp /= h_out;
    const int channel = tmp % c;
    const int batch = tmp / c;

    const float in_y = align_corners && h_out > 1
        ? static_cast<float>(oh) * static_cast<float>(h_in - 1) / static_cast<float>(h_out - 1)
        : (static_cast<float>(oh) + 0.5f) * static_cast<float>(h_in) / static_cast<float>(h_out) - 0.5f;
    const float in_x = align_corners && w_out > 1
        ? static_cast<float>(ow) * static_cast<float>(w_in - 1) / static_cast<float>(w_out - 1)
        : (static_cast<float>(ow) + 0.5f) * static_cast<float>(w_in) / static_cast<float>(w_out) - 0.5f;

    const float y = fminf(fmaxf(in_y, 0.0f), static_cast<float>(h_in - 1));
    const float x = fminf(fmaxf(in_x, 0.0f), static_cast<float>(w_in - 1));
    const int y0 = static_cast<int>(floorf(y));
    const int x0 = static_cast<int>(floorf(x));
    const int y1 = min(y0 + 1, h_in - 1);
    const int x1 = min(x0 + 1, w_in - 1);
    const float ly = y - static_cast<float>(y0);
    const float lx = x - static_cast<float>(x0);

    const int base = (batch * c + channel) * h_in;
    const float v00 = input[(base + y0) * w_in + x0];
    const float v01 = input[(base + y0) * w_in + x1];
    const float v10 = input[(base + y1) * w_in + x0];
    const float v11 = input[(base + y1) * w_in + x1];
    const float top = v00 * (1.0f - lx) + v01 * lx;
    const float bottom = v10 * (1.0f - lx) + v11 * lx;
    output[linear] = top * (1.0f - ly) + bottom * ly;
}

bool invalid_args(const float* input, const float* output, const int n, const int c, const int h_in, const int w_in,
    const int h_out, const int w_out) {
    return input == nullptr || output == nullptr || n <= 0 || c <= 0 || h_in <= 0 || w_in <= 0 || h_out <= 0 ||
        w_out <= 0;
}

}  // namespace

cudaError_t upsample_nearest2d_nchw(
    const float* input,
    float* output,
    const int n,
    const int c,
    const int h_in,
    const int w_in,
    const int h_out,
    const int w_out,
    cudaStream_t stream) {
    if (invalid_args(input, output, n, c, h_in, w_in, h_out, w_out)) {
        return cudaErrorInvalidValue;
    }
    const int total = n * c * h_out * w_out;
    const int blocks = (total + kBlockSize - 1) / kBlockSize;
    upsample_nearest2d_nchw_kernel<<<blocks, kBlockSize, 0, stream>>>(input, output, c, h_in, w_in, h_out, w_out, total);
    return cudaGetLastError();
}

cudaError_t upsample_bilinear2d_nchw(
    const float* input,
    float* output,
    const int n,
    const int c,
    const int h_in,
    const int w_in,
    const int h_out,
    const int w_out,
    const bool align_corners,
    cudaStream_t stream) {
    if (invalid_args(input, output, n, c, h_in, w_in, h_out, w_out)) {
        return cudaErrorInvalidValue;
    }
    const int total = n * c * h_out * w_out;
    const int blocks = (total + kBlockSize - 1) / kBlockSize;
    upsample_bilinear2d_nchw_kernel<<<blocks, kBlockSize, 0, stream>>>(
        input,
        output,
        c,
        h_in,
        w_in,
        h_out,
        w_out,
        align_corners,
        total);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
