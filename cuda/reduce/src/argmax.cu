#include "wgkernel/cuda/reduce.hpp"

#include "reduce_utils.cuh"

#include <cfloat>
#include <cstdint>

namespace wgkernel::cuda {
namespace {

constexpr float kNegativeInfinity = -FLT_MAX;
constexpr std::int64_t kInvalidIndex = 0x7fffffffffffffffLL;

struct ArgMaxPair {
    float value;
    std::int64_t index;
};

struct ArgMaxReducer {
    __device__ ArgMaxPair operator()(const ArgMaxPair& lhs, const ArgMaxPair& rhs) const {
        if (lhs.value > rhs.value) {
            return lhs;
        }
        if (lhs.value < rhs.value) {
            return rhs;
        }
        return lhs.index <= rhs.index ? lhs : rhs;
    }
};

__global__ void reduce_argmax_v1_first_stage_kernel(
    const float* input,
    ArgMaxPair* output,
    const std::int64_t numel) {
    extern __shared__ __align__(16) unsigned char shared_memory[];
    auto* scratch = reinterpret_cast<ArgMaxPair*>(shared_memory);

    const int thread_id = threadIdx.x;
    const std::int64_t global_thread_id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

    ArgMaxReducer reducer;
    ArgMaxPair accumulator { kNegativeInfinity, kInvalidIndex };

    for (std::int64_t index = global_thread_id; index < numel; index += stride) {
        const ArgMaxPair candidate { input[index], index };
        accumulator = reducer(accumulator, candidate);
    }

    const ArgMaxPair block_argmax = reduce_detail::block_reduce(accumulator, scratch, reducer);
    if (thread_id == 0) {
        output[blockIdx.x] = block_argmax;
    }
}

__global__ void reduce_argmax_v1_pair_stage_kernel(
    const ArgMaxPair* input,
    ArgMaxPair* output,
    const std::int64_t numel) {
    extern __shared__ __align__(16) unsigned char shared_memory[];
    auto* scratch = reinterpret_cast<ArgMaxPair*>(shared_memory);

    const int thread_id = threadIdx.x;
    const std::int64_t global_thread_id = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + thread_id;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;

    ArgMaxReducer reducer;
    ArgMaxPair accumulator { kNegativeInfinity, kInvalidIndex };

    for (std::int64_t index = global_thread_id; index < numel; index += stride) {
        accumulator = reducer(accumulator, input[index]);
    }

    const ArgMaxPair block_argmax = reduce_detail::block_reduce(accumulator, scratch, reducer);
    if (thread_id == 0) {
        output[blockIdx.x] = block_argmax;
    }
}

__global__ void reduce_argmax_v1_extract_index_kernel(const ArgMaxPair* input, std::int64_t* output_index) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output_index[0] = input[0].index;
    }
}

cudaError_t launch_reduce_argmax_v1_first_stage(
    const float* input,
    ArgMaxPair* output,
    const std::int64_t numel,
    const int blocks,
    cudaStream_t stream) {
    reduce_argmax_v1_first_stage_kernel
        <<<blocks, reduce_detail::kBlockSize, sizeof(ArgMaxPair) * reduce_detail::kBlockSize, stream>>>(
            input,
            output,
            numel);
    return cudaGetLastError();
}

cudaError_t launch_reduce_argmax_v1_pair_stage(
    const ArgMaxPair* input,
    ArgMaxPair* output,
    const std::int64_t numel,
    const int blocks,
    cudaStream_t stream) {
    reduce_argmax_v1_pair_stage_kernel
        <<<blocks, reduce_detail::kBlockSize, sizeof(ArgMaxPair) * reduce_detail::kBlockSize, stream>>>(
            input,
            output,
            numel);
    return cudaGetLastError();
}

}  // namespace

std::size_t reduce_argmax_workspace_size(const std::int64_t numel) {
    if (numel <= 0) {
        return 0;
    }

    const int initial_blocks = reduce_detail::compute_launch_blocks(numel);
    const std::size_t buffers = initial_blocks > 1 ? static_cast<std::size_t>(initial_blocks) * 2U : 1U;
    return buffers * sizeof(ArgMaxPair);
}

cudaError_t reduce_argmax(
    const float* input,
    std::int64_t* output_index,
    const std::int64_t numel,
    void* workspace,
    const std::size_t workspace_bytes,
    cudaStream_t stream) {
    if (input == nullptr || output_index == nullptr || numel <= 0 || workspace == nullptr) {
        return cudaErrorInvalidValue;
    }

    const std::size_t required_workspace = reduce_argmax_workspace_size(numel);
    if (workspace_bytes < required_workspace) {
        return cudaErrorInvalidValue;
    }

    const int initial_blocks = reduce_detail::compute_launch_blocks(numel);
    auto* first_buffer = static_cast<ArgMaxPair*>(workspace);
    auto* second_buffer = initial_blocks > 1 ? first_buffer + initial_blocks : first_buffer;

    cudaError_t status = launch_reduce_argmax_v1_first_stage(input, first_buffer, numel, initial_blocks, stream);
    if (status != cudaSuccess) {
        return status;
    }

    ArgMaxPair* current = first_buffer;
    ArgMaxPair* next = second_buffer;
    std::int64_t current_numel = initial_blocks;

    while (current_numel > 1) {
        const int next_blocks = reduce_detail::compute_launch_blocks(current_numel);
        status = launch_reduce_argmax_v1_pair_stage(current, next, current_numel, next_blocks, stream);
        if (status != cudaSuccess) {
            return status;
        }

        current = next;
        next = (next == first_buffer) ? second_buffer : first_buffer;
        current_numel = next_blocks;
    }

    reduce_argmax_v1_extract_index_kernel<<<1, 1, 0, stream>>>(current, output_index);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
