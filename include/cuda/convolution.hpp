#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace wgkernel::cuda {

// Algorithm 1: naive direct convolution, one thread per output element.
cudaError_t conv2d_nchw(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    cudaStream_t stream = nullptr);

// Algorithm 2: im2col + GEMM. Unfolds the input into a column matrix and runs a
// tiled GEMM against the reshaped weights. Allocates scratch internally.
cudaError_t conv2d_nchw_im2col_gemm(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    cudaStream_t stream = nullptr);

// Algorithm 3: direct tiled convolution. Each block computes an output tile and
// stages the corresponding input halo in shared memory for reuse.
cudaError_t conv2d_nchw_direct_tiled(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    cudaStream_t stream = nullptr);

// Algorithm 4: implicit GEMM. Same tiling as algorithm 2 but the column matrix
// is never materialized; B tiles are gathered from the input on the fly.
cudaError_t conv2d_nchw_implicit_gemm(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int n,
    int c_in,
    int h_in,
    int w_in,
    int c_out,
    int k_h,
    int k_w,
    int h_out,
    int w_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    cudaStream_t stream = nullptr);

}  // namespace wgkernel::cuda
