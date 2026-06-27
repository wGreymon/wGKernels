#include "cpu/embedding_indexing.hpp"

#include <cstdint>
#include <stdexcept>

namespace wgkernel::cpu {
namespace {

void check_embedding_args(
    const float* weight,
    const std::int64_t* indices,
    const float* output,
    const std::int64_t num_indices,
    const std::int64_t num_embeddings,
    const std::int64_t embedding_dim) {
    if (weight == nullptr || indices == nullptr || output == nullptr) {
        throw std::invalid_argument("weight, indices and output must not be null");
    }
    if (num_indices <= 0 || num_embeddings <= 0 || embedding_dim <= 0) {
        throw std::invalid_argument("embedding dimensions must be positive");
    }
}

}  // namespace

void embedding(
    const float* weight,
    const std::int64_t* indices,
    float* output,
    const std::int64_t num_indices,
    const std::int64_t num_embeddings,
    const std::int64_t embedding_dim) {
    check_embedding_args(weight, indices, output, num_indices, num_embeddings, embedding_dim);

    for (std::int64_t index_pos = 0; index_pos < num_indices; ++index_pos) {
        const std::int64_t embedding_index = indices[index_pos];
        if (embedding_index < 0 || embedding_index >= num_embeddings) {
            throw std::out_of_range("embedding index is out of range");
        }

        const std::int64_t input_offset = embedding_index * embedding_dim;
        const std::int64_t output_offset = index_pos * embedding_dim;
        for (std::int64_t dim = 0; dim < embedding_dim; ++dim) {
            output[output_offset + dim] = weight[input_offset + dim];
        }
    }
}

}  // namespace wgkernel::cpu
