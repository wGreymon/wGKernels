#pragma once

#include <cstdint>

namespace wgkernel::cpu {

void embedding(
    const float* weight,
    const std::int64_t* indices,
    float* output,
    std::int64_t num_indices,
    std::int64_t num_embeddings,
    std::int64_t embedding_dim);

}  // namespace wgkernel::cpu
