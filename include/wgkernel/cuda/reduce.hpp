#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace wgkernel::cuda {

std::size_t reduce_sum_workspace_size(std::int64_t numel);
std::size_t reduce_max_workspace_size(std::int64_t numel);
std::size_t reduce_argmax_workspace_size(std::int64_t numel);

cudaError_t reduce_sum(
    const float* input,
    float* output,
    std::int64_t numel,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream = nullptr);

cudaError_t reduce_max(
    const float* input,
    float* output,
    std::int64_t numel,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream = nullptr);

cudaError_t reduce_argmax(
    const float* input,
    std::int64_t* output_index,
    std::int64_t numel,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
