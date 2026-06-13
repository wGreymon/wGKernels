#include "cuda/pooling.hpp"

#include <cuda_runtime.h>

#include <cfloat>

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;

__global__ void maxpool2d_nchw_kernel(
    const float* input,
    float* output,
    const int c,
    const int h_in,
    const int w_in,
    const int h_out,
    const int w_out,
    const int k_h,
    const int k_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
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

    float max_value = -FLT_MAX;
    for (int kh = 0; kh < k_h; ++kh) {
        const int ih = oh * stride_h - pad_h + kh;
        if (ih < 0 || ih >= h_in) {
            continue;
        }
        for (int kw = 0; kw < k_w; ++kw) {
            const int iw = ow * stride_w - pad_w + kw;
            if (iw < 0 || iw >= w_in) {
                continue;
            }
            const int input_index = ((batch * c + channel) * h_in + ih) * w_in + iw;
            max_value = fmaxf(max_value, input[input_index]);
        }
    }
    output[linear] = max_value;
}

}  // namespace

cudaError_t maxpool2d_nchw(
    const float* input,
    float* output,
    const int n,
    const int c,
    const int h_in,
    const int w_in,
    const int h_out,
    const int w_out,
    const int k_h,
    const int k_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || n <= 0 || c <= 0 || h_in <= 0 || w_in <= 0 || h_out <= 0 ||
        w_out <= 0 || k_h <= 0 || k_w <= 0 || stride_h <= 0 || stride_w <= 0) {
        return cudaErrorInvalidValue;
    }

    const int total = n * c * h_out * w_out;
    const int blocks = (total + kBlockSize - 1) / kBlockSize;
    maxpool2d_nchw_kernel<<<blocks, kBlockSize, 0, stream>>>(
        input,
        output,
        c,
        h_in,
        w_in,
        h_out,
        w_out,
        k_h,
        k_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        total);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
