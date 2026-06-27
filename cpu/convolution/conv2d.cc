#include "cpu/convolution.hpp"

#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_conv2d_args(
    const float* input,
    const float* weight,
    const float* output,
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
    const int dilation_h,
    const int dilation_w,
    const int groups) {
    if (input == nullptr || weight == nullptr || output == nullptr) {
        throw std::invalid_argument("input, weight and output must not be null");
    }
    if (n <= 0 || c_in <= 0 || h_in <= 0 || w_in <= 0 || c_out <= 0 || k_h <= 0 || k_w <= 0 ||
        h_out <= 0 || w_out <= 0 || stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 ||
        groups <= 0) {
        throw std::invalid_argument("conv2d dimensions must be positive");
    }
    if (c_in % groups != 0 || c_out % groups != 0) {
        throw std::invalid_argument("c_in and c_out must be divisible by groups");
    }
}

}  // namespace

void conv2d_nchw(
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
    const int groups) {
    check_conv2d_args(
        input,
        weight,
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
        dilation_h,
        dilation_w,
        groups);

    const int c_in_per_group = c_in / groups;
    const int c_out_per_group = c_out / groups;

    for (int ni = 0; ni < n; ++ni) {
        for (int co = 0; co < c_out; ++co) {
            const int group = co / c_out_per_group;
            const int c_in_begin = group * c_in_per_group;
            for (int ho = 0; ho < h_out; ++ho) {
                for (int wo = 0; wo < w_out; ++wo) {
                    float accumulator = bias != nullptr ? bias[co] : 0.0F;
                    for (int ci_group = 0; ci_group < c_in_per_group; ++ci_group) {
                        const int ci = c_in_begin + ci_group;
                        for (int kh = 0; kh < k_h; ++kh) {
                            const int hi = ho * stride_h - pad_h + kh * dilation_h;
                            if (hi < 0 || hi >= h_in) {
                                continue;
                            }
                            for (int kw = 0; kw < k_w; ++kw) {
                                const int wi = wo * stride_w - pad_w + kw * dilation_w;
                                if (wi < 0 || wi >= w_in) {
                                    continue;
                                }

                                const int input_index = ((ni * c_in + ci) * h_in + hi) * w_in + wi;
                                const int weight_index = ((co * c_in_per_group + ci_group) * k_h + kh) * k_w + kw;
                                accumulator += input[input_index] * weight[weight_index];
                            }
                        }
                    }
                    const int output_index = ((ni * c_out + co) * h_out + ho) * w_out + wo;
                    output[output_index] = accumulator;
                }
            }
        }
    }
}

}  // namespace wgkernel::cpu
