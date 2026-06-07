#include "wgkernel/cuda/embedding_indexing.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;

int launch_blocks(const std::int64_t numel) {
    return static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
}

__global__ void slice_1d_kernel(
    const float* input,
    float* output,
    const std::int64_t input_numel,
    const std::int64_t start,
    const std::int64_t step,
    const std::int64_t output_numel) {
    const std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_numel) {
        return;
    }
    const std::int64_t input_index = start + index * step;
    output[index] = input_index >= 0 && input_index < input_numel ? input[input_index] : 0.0f;
}

__global__ void gather_1d_kernel(
    const float* input,
    const std::int64_t* indices,
    float* output,
    const std::int64_t output_numel) {
    const std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < output_numel) {
        output[index] = input[indices[index]];
    }
}

__global__ void topk_1d_kernel(
    const float* input,
    float* values,
    std::int64_t* indices,
    const std::int64_t numel,
    const int k) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (int out = 0; out < k; ++out) {
        float best_value = -FLT_MAX;
        std::int64_t best_index = -1;
        for (std::int64_t index = 0; index < numel; ++index) {
            bool used = false;
            for (int prev = 0; prev < out; ++prev) {
                used = used || indices[prev] == index;
            }
            if (!used && (input[index] > best_value || (input[index] == best_value && index < best_index))) {
                best_value = input[index];
                best_index = index;
            }
        }
        values[out] = best_value;
        indices[out] = best_index;
    }
}

__global__ void sort_1d_kernel(
    const float* input,
    float* values,
    std::int64_t* indices,
    const std::int64_t numel,
    const bool descending) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (std::int64_t index = 0; index < numel; ++index) {
        values[index] = input[index];
        indices[index] = index;
    }

    for (std::int64_t i = 0; i < numel; ++i) {
        for (std::int64_t j = i + 1; j < numel; ++j) {
            const bool should_swap = descending ? values[j] > values[i] : values[j] < values[i];
            if (should_swap) {
                const float value = values[i];
                values[i] = values[j];
                values[j] = value;
                const std::int64_t index = indices[i];
                indices[i] = indices[j];
                indices[j] = index;
            }
        }
    }
}

}  // namespace

cudaError_t slice_1d(
    const float* input,
    float* output,
    const std::int64_t input_numel,
    const std::int64_t start,
    const std::int64_t step,
    const std::int64_t output_numel,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || input_numel <= 0 || step == 0 || output_numel <= 0) {
        return cudaErrorInvalidValue;
    }
    slice_1d_kernel<<<launch_blocks(output_numel), kBlockSize, 0, stream>>>(input, output, input_numel, start, step, output_numel);
    return cudaGetLastError();
}

cudaError_t gather_1d(
    const float* input,
    const std::int64_t* indices,
    float* output,
    const std::int64_t output_numel,
    cudaStream_t stream) {
    if (input == nullptr || indices == nullptr || output == nullptr || output_numel <= 0) {
        return cudaErrorInvalidValue;
    }
    gather_1d_kernel<<<launch_blocks(output_numel), kBlockSize, 0, stream>>>(input, indices, output, output_numel);
    return cudaGetLastError();
}

cudaError_t topk_1d(
    const float* input,
    float* values,
    std::int64_t* indices,
    const std::int64_t numel,
    const int k,
    cudaStream_t stream) {
    if (input == nullptr || values == nullptr || indices == nullptr || numel <= 0 || k <= 0 || k > numel) {
        return cudaErrorInvalidValue;
    }
    topk_1d_kernel<<<1, 1, 0, stream>>>(input, values, indices, numel, k);
    return cudaGetLastError();
}

cudaError_t sort_1d(
    const float* input,
    float* values,
    std::int64_t* indices,
    const std::int64_t numel,
    const bool descending,
    cudaStream_t stream) {
    if (input == nullptr || values == nullptr || indices == nullptr || numel <= 0) {
        return cudaErrorInvalidValue;
    }
    sort_1d_kernel<<<1, 1, 0, stream>>>(input, values, indices, numel, descending);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
