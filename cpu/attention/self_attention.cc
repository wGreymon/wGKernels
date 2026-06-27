#include "cpu/attention.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace wgkernel::cpu {
namespace {

void check_attention_args(
    const float* query,
    const float* key,
    const float* value,
    const float* output,
    const std::int64_t batch,
    const std::int64_t query_length,
    const std::int64_t key_value_length,
    const std::int64_t head_dim,
    const std::int64_t value_dim) {
    if (query == nullptr || key == nullptr || value == nullptr || output == nullptr) {
        throw std::invalid_argument("query, key, value and output must not be null");
    }
    if (batch <= 0 || query_length <= 0 || key_value_length <= 0 || head_dim <= 0 || value_dim <= 0) {
        throw std::invalid_argument("attention dimensions must be positive");
    }
}

}  // namespace

void self_attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    const std::int64_t batch,
    const std::int64_t query_length,
    const std::int64_t key_value_length,
    const std::int64_t head_dim,
    const std::int64_t value_dim,
    const bool causal) {
    check_attention_args(query, key, value, output, batch, query_length, key_value_length, head_dim, value_dim);

    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<std::size_t>(key_value_length));

    for (std::int64_t b = 0; b < batch; ++b) {
        const float* q_batch = query + b * query_length * head_dim;
        const float* k_batch = key + b * key_value_length * head_dim;
        const float* v_batch = value + b * key_value_length * value_dim;
        float* out_batch = output + b * query_length * value_dim;

        for (std::int64_t qi = 0; qi < query_length; ++qi) {
            float max_score = -std::numeric_limits<float>::infinity();
            for (std::int64_t ki = 0; ki < key_value_length; ++ki) {
                float score = -std::numeric_limits<float>::infinity();
                if (!causal || ki <= qi) {
                    score = 0.0F;
                    for (std::int64_t d = 0; d < head_dim; ++d) {
                        score += q_batch[qi * head_dim + d] * k_batch[ki * head_dim + d];
                    }
                    score *= scale;
                }
                scores[static_cast<std::size_t>(ki)] = score;
                max_score = max_score > score ? max_score : score;
            }

            float denominator = 0.0F;
            for (std::int64_t ki = 0; ki < key_value_length; ++ki) {
                const float weight = std::exp(scores[static_cast<std::size_t>(ki)] - max_score);
                scores[static_cast<std::size_t>(ki)] = weight;
                denominator += weight;
            }

            for (std::int64_t vd = 0; vd < value_dim; ++vd) {
                float accumulator = 0.0F;
                for (std::int64_t ki = 0; ki < key_value_length; ++ki) {
                    const float weight = scores[static_cast<std::size_t>(ki)] / denominator;
                    accumulator += weight * v_batch[ki * value_dim + vd];
                }
                out_batch[qi * value_dim + vd] = accumulator;
            }
        }
    }
}

}  // namespace wgkernel::cpu
