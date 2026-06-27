#include "cpu/pooling.hpp"

#include <cfloat>
#include <cstdint>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_maxpool_args(
    const float* input,
    const float* output,
    const std::int64_t n,
    const std::int64_t c,
    const std::int64_t h_in,
    const std::int64_t w_in,
    const std::int64_t h_out,
    const std::int64_t w_out,
    const std::int64_t k_h,
    const std::int64_t k_w,
    const std::int64_t stride_h,
    const std::int64_t stride_w) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("input and output must not be null");
    }
    if (n <= 0 || c <= 0 || h_in <= 0 || w_in <= 0 || h_out <= 0 || w_out <= 0 || k_h <= 0 || k_w <= 0 ||
        stride_h <= 0 || stride_w <= 0) {
        throw std::invalid_argument("maxpool dimensions, kernel size and stride must be positive");
    }
}

}  // namespace

void maxpool2d_nchw(
    const float* input,
    float* output,
    const std::int64_t n,
    const std::int64_t c,
    const std::int64_t h_in,
    const std::int64_t w_in,
    const std::int64_t h_out,
    const std::int64_t w_out,
    const std::int64_t k_h,
    const std::int64_t k_w,
    const std::int64_t stride_h,
    const std::int64_t stride_w,
    const std::int64_t pad_h,
    const std::int64_t pad_w) {
    check_maxpool_args(input, output, n, c, h_in, w_in, h_out, w_out, k_h, k_w, stride_h, stride_w);

    for (std::int64_t batch = 0; batch < n; ++batch) {
        for (std::int64_t channel = 0; channel < c; ++channel) {
            for (std::int64_t oh = 0; oh < h_out; ++oh) {
                for (std::int64_t ow = 0; ow < w_out; ++ow) {
                    float max_value = -FLT_MAX;
                    for (std::int64_t kh = 0; kh < k_h; ++kh) {
                        const std::int64_t ih = oh * stride_h - pad_h + kh;
                        if (ih < 0 || ih >= h_in) {
                            continue;
                        }
                        for (std::int64_t kw = 0; kw < k_w; ++kw) {
                            const std::int64_t iw = ow * stride_w - pad_w + kw;
                            if (iw < 0 || iw >= w_in) {
                                continue;
                            }
                            const std::int64_t input_index = ((batch * c + channel) * h_in + ih) * w_in + iw;
                            max_value = input[input_index] > max_value ? input[input_index] : max_value;
                        }
                    }
                    const std::int64_t output_index = ((batch * c + channel) * h_out + oh) * w_out + ow;
                    output[output_index] = max_value;
                }
            }
        }
    }
}

}  // namespace wgkernel::cpu
