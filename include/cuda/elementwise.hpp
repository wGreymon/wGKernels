#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace wgkernel::cuda {

cudaError_t add(const float* a, const float* b, float* output, std::int64_t numel, cudaStream_t stream = nullptr);
cudaError_t sub(const float* a, const float* b, float* output, std::int64_t numel, cudaStream_t stream = nullptr);
cudaError_t mul(const float* a, const float* b, float* output, std::int64_t numel, cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
