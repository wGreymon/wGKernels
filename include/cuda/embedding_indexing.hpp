#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace wgkernel::cuda {

cudaError_t slice_1d(
    const float* input,
    float* output,
    std::int64_t input_numel,
    std::int64_t start,
    std::int64_t step,
    std::int64_t output_numel,
    cudaStream_t stream = nullptr);

cudaError_t gather_1d(
    const float* input,
    const std::int64_t* indices,
    float* output,
    std::int64_t output_numel,
    cudaStream_t stream = nullptr);

cudaError_t topk_1d(
    const float* input,
    float* values,
    std::int64_t* indices,
    std::int64_t numel,
    int k,
    cudaStream_t stream = nullptr);

cudaError_t sort_1d(
    const float* input,
    float* values,
    std::int64_t* indices,
    std::int64_t numel,
    bool descending,
    cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
