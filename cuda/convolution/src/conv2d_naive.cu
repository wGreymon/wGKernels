// 算法 1：朴素直接卷积。
//
// 一个线程负责一个输出元素 output[n][oc][oh][ow]，自己走完整个
// (in_channel, k_h, k_w) 的累加。简单且正确，但每个线程都从 global memory
// 重复读取 weight 和 input，没有任何复用。

#include "cuda/convolution.hpp"

#include <cuda_runtime.h>

#include "conv2d_internal.cuh"

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;  // elementwise kernel 每个 block 的线程数

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
    if (conv2d_internal::conv2d_args_invalid(
            input, weight, output, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
            stride_h, stride_w, dilation_h, dilation_w, groups)) {
        return cudaErrorInvalidValue;
    }

    const int total = n * c_out * h_out * w_out;
    const int blocks = (total + kBlockSize - 1) / kBlockSize;
    conv2d_nchw_kernel<<<blocks, kBlockSize, 0, stream>>>(
        input, weight, bias, output, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, total);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
