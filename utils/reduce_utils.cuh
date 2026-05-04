#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace wgkernel::cuda::reduce_detail {

// A reasonable default for the first round of reduce kernels on recent GPUs.
constexpr int kBlockSize = 256;
constexpr int kTargetItemsPerThread = 8;
constexpr int kMaxBlocks = 1024;
constexpr int kWarpSize = 32;
constexpr unsigned kWarpFullMask = 0xffffffffu;

inline std::int64_t ceil_div(const std::int64_t value, const std::int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline int compute_launch_blocks(const std::int64_t numel) {
    const std::int64_t target_items_per_block = static_cast<std::int64_t>(kBlockSize) * kTargetItemsPerThread;
    const std::int64_t suggested = ceil_div(numel, target_items_per_block);

    if (suggested < 1) {
        return 1;
    }
    if (suggested > kMaxBlocks) {
        return kMaxBlocks;
    }
    return static_cast<int>(suggested);
}

template <typename T>
std::size_t scalar_reduce_workspace_size(const std::int64_t numel) {
    if (numel <= 0) {
        return 0;
    }

    const int initial_blocks = compute_launch_blocks(numel);
    if (initial_blocks == 1) {
        return 0;
    }

    return static_cast<std::size_t>(initial_blocks) * 2U * sizeof(T);
}

template <typename T, typename Reducer>
__device__ T block_reduce(T value, T* shared_data, const Reducer reducer) {
    const int thread_id = threadIdx.x;

    shared_data[thread_id] = value;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (thread_id < offset) {
            shared_data[thread_id] = reducer(shared_data[thread_id], shared_data[thread_id + offset]);
        }
        __syncthreads();
    }

    return shared_data[0];
}

template <typename T, typename Reducer>
__device__ T warp_reduce(T value, const Reducer reducer) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = reducer(value, __shfl_down_sync(kWarpFullMask, value, offset));
    }
    return value;
}

}  // namespace wgkernel::cuda::reduce_detail
