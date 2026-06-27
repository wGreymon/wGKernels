#pragma once

#include <cstdint>

namespace wgkernel::cpu {

float reduce_sum(const float* input, std::int64_t numel);
float reduce_max(const float* input, std::int64_t numel);
std::int64_t reduce_argmax(const float* input, std::int64_t numel);
void softmax(const float* input, float* output, std::int64_t numel);

}  // namespace wgkernel::cpu
