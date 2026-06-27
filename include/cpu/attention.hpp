#pragma once

#include <cstdint>

namespace wgkernel::cpu {

void self_attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    std::int64_t batch,
    std::int64_t query_length,
    std::int64_t key_value_length,
    std::int64_t head_dim,
    std::int64_t value_dim,
    bool causal);

}  // namespace wgkernel::cpu
