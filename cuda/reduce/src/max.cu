#include "cuda/reduce.hpp"

#include "reduce_utils.cuh"

#include <cfloat>

namespace wgkernel::cuda {
namespace {

struct MaxReducer {
    __device__ float operator()(const float lhs, const float rhs) const {
        return lhs > rhs ? lhs : rhs;
    }
};

// v1: full block reduction through shared memory.
__global__ void reduce_max_v1_kernel(const float* input, float* output, const std::int64_t numel) {
    extern __shared__ float shared_data[];

    const int thread_id = threadIdx.x;
    const std::int64_t global_thread_id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + thread_id;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

    float max_value = -FLT_MAX;
    for (std::int64_t index = global_thread_id; index < numel; index += stride) {
        max_value = max_value > input[index] ? max_value : input[index];
    }

    const float block_max = reduce_detail::block_reduce(max_value, shared_data, MaxReducer {});
    if (thread_id == 0) {
        output[blockIdx.x] = block_max;
    }
}

// v2: use warp shuffle within each warp, then reduce warp results.
__global__ void reduce_max_v2_kernel(const float* input, float* output, const std::int64_t numel) {
    __shared__ float warp_max_values[reduce_detail::kBlockSize / reduce_detail::kWarpSize];

    const int thread_id = threadIdx.x;
    const int lane_id = thread_id & (reduce_detail::kWarpSize - 1);
    const int warp_id = thread_id / reduce_detail::kWarpSize;
    const int num_warps = blockDim.x / reduce_detail::kWarpSize;
    const std::int64_t global_thread_id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + thread_id;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

    float max_value = -FLT_MAX;
    for (std::int64_t index = global_thread_id; index < numel; index += stride) {
        max_value = max_value > input[index] ? max_value : input[index];
    }

    max_value = reduce_detail::warp_reduce(max_value, MaxReducer {});
    if (lane_id == 0) {
        warp_max_values[warp_id] = max_value;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = lane_id < num_warps ? warp_max_values[lane_id] : -FLT_MAX;
        block_max = reduce_detail::warp_reduce(block_max, MaxReducer {});
        if (lane_id == 0) {
            output[blockIdx.x] = block_max;
        }
    }
}

[[maybe_unused]] cudaError_t launch_reduce_max_v1(
    const float* input,
    float* output,
    const std::int64_t numel,
    const int blocks,
    cudaStream_t stream) {
    reduce_max_v1_kernel<<<blocks, reduce_detail::kBlockSize, sizeof(float) * reduce_detail::kBlockSize, stream>>>(
        input,
        output,
        numel);
    return cudaGetLastError();
}

cudaError_t launch_reduce_max_v2(
    const float* input,
    float* output,
    const std::int64_t numel,
    const int blocks,
    cudaStream_t stream) {
    reduce_max_v2_kernel<<<blocks, reduce_detail::kBlockSize, 0, stream>>>(
        input,
        output,
        numel);
    return cudaGetLastError();
}

}  // namespace

std::size_t reduce_max_workspace_size(const std::int64_t numel) {
    return reduce_detail::scalar_reduce_workspace_size<float>(numel);
}

cudaError_t reduce_max(
    const float* input,
    float* output,
    const std::int64_t numel,
    void* workspace,
    const std::size_t workspace_bytes,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || numel <= 0) {
        return cudaErrorInvalidValue;
    }

    const int initial_blocks = reduce_detail::compute_launch_blocks(numel);
    if (initial_blocks == 1) {
        return launch_reduce_max_v2(input, output, numel, 1, stream);
    }

    const std::size_t required_workspace = reduce_max_workspace_size(numel);
    if (workspace == nullptr || workspace_bytes < required_workspace) {
        return cudaErrorInvalidValue;
    }

    auto* first_buffer = static_cast<float*>(workspace);
    auto* second_buffer = first_buffer + initial_blocks;

    cudaError_t status = launch_reduce_max_v2(input, first_buffer, numel, initial_blocks, stream);
    if (status != cudaSuccess) {
        return status;
    }

    float* current = first_buffer;
    float* next = second_buffer;
    std::int64_t current_numel = initial_blocks;

    while (current_numel > 1) {
        const int blocks = reduce_detail::compute_launch_blocks(current_numel);
        status = launch_reduce_max_v2(current, next, current_numel, blocks, stream);
        if (status != cudaSuccess) {
            return status;
        }

        current = next;
        next = (next == first_buffer) ? second_buffer : first_buffer;
        current_numel = blocks;
    }

    return cudaMemcpyAsync(output, current, sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

}  // namespace wgkernel::cuda
