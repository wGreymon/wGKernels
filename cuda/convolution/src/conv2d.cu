#include "wgkernel/cuda/convolution.hpp"

#include <cuda_runtime.h>

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;

__global__ void conv2d_nchw_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int n,
    const int c_in,
    const int h_in,
    const int w_in,
    const int c_out,
    const int k_h,
    const int k_w,
    const int h_out,
    const int w_out,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
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
    const int oc = tmp % c_out;
    const int batch = tmp / c_out;

    const int out_channels_per_group = c_out / groups;
    const int in_channels_per_group = c_in / groups;
    const int group = oc / out_channels_per_group;
    const int ic_begin = group * in_channels_per_group;

    float acc = bias == nullptr ? 0.0f : bias[oc];

    for (int icg = 0; icg < in_channels_per_group; ++icg) {
        const int ic = ic_begin + icg;
        for (int kh = 0; kh < k_h; ++kh) {
            const int ih = oh * stride_h - pad_h + kh * dilation_h;
            if (ih < 0 || ih >= h_in) {
                continue;
            }
            for (int kw = 0; kw < k_w; ++kw) {
                const int iw = ow * stride_w - pad_w + kw * dilation_w;
                if (iw < 0 || iw >= w_in) {
                    continue;
                }
                const int input_index = ((batch * c_in + ic) * h_in + ih) * w_in + iw;
                const int weight_index = ((oc * in_channels_per_group + icg) * k_h + kh) * k_w + kw;
                acc += input[input_index] * weight[weight_index];
            }
        }
    }

    output[linear] = acc;
}

}  // namespace

cudaError_t conv2d_nchw(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int n,
    const int c_in,
    const int h_in,
    const int w_in,
    const int c_out,
    const int k_h,
    const int k_w,
    const int h_out,
    const int w_out,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    cudaStream_t stream) {
    if (input == nullptr || weight == nullptr || output == nullptr || n <= 0 || c_in <= 0 || h_in <= 0 ||
        w_in <= 0 || c_out <= 0 || k_h <= 0 || k_w <= 0 || h_out <= 0 || w_out <= 0 || stride_h <= 0 ||
        stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 || groups <= 0 || c_in % groups != 0 ||
        c_out % groups != 0) {
        return cudaErrorInvalidValue;
    }

    const int total = n * c_out * h_out * w_out;
    const int blocks = (total + kBlockSize - 1) / kBlockSize;
    conv2d_nchw_kernel<<<blocks, kBlockSize, 0, stream>>>(
        input,
        weight,
        bias,
        output,
        n,
        c_in,
        h_in,
        w_in,
        c_out,
        k_h,
        k_w,
        h_out,
        w_out,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        total);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
