// 算法 2：im2col + GEMM。
//
// 对每个 (batch, group)，卷积变成一次矩阵乘：
//
//   C[M, N] = W[M, K] * Col[K, N]   （bias 沿 N 维广播）
//
//   M = out_channels_per_group
//   K = in_channels_per_group * k_h * k_w   （归约维度）
//   N = h_out * w_out                       （输出空间位置数）
//
// im2col_kernel 把 Col 物化出来：每一列 p = (oh, ow) 存放对应感受野按
// (icg, kh, kw) 展平后的输入，padding 处填 0。某个 (batch, group) 对应的
// weight 和 output 子块本身就是连续 row-major 的，所以 GEMM 直接带 offset 读写。

#include "cuda/convolution.hpp"

#include <cuda_runtime.h>

#include <cstddef>

#include "conv2d_internal.cuh"

namespace wgkernel::cuda {
namespace {

constexpr int kTile = 16;  // GEMM tile 边长

// 为一个 (batch, group) 填充列矩阵 Col[K, N]。
//   行下标 k -> (icg, kh, kw)
//   列下标 p -> (oh, ow)
__global__ void im2col_kernel(
    const float* input,
    float* col,
    const int batch,
    const int ic_begin,
    const int c_in,
    const int h_in,
    const int w_in,
    const int in_channels_per_group,
    const int k_h,
    const int k_w,
    const int h_out,
    const int w_out,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int k_dim,
    const int n_dim) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;  // 输出空间下标
    const int k = blockIdx.y * blockDim.y + threadIdx.y;  // 归约下标
    if (p >= n_dim || k >= k_dim) {
        return;
    }

    const int kw = k % k_w;
    int tmp = k / k_w;
    const int kh = tmp % k_h;
    const int icg = tmp / k_h;
    const int ic = ic_begin + icg;

    const int ow = p % w_out;
    const int oh = p / w_out;

    const int ih = oh * stride_h - pad_h + kh * dilation_h;
    const int iw = ow * stride_w - pad_w + kw * dilation_w;

    float value = 0.0f;
    if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
        value = input[((batch * c_in + ic) * h_in + ih) * w_in + iw];
    }
    col[k * n_dim + p] = value;
}

// Tiled GEMM：C[M, N] = A[M, K] * B[K, N]，可选 per-row bias。
// A、B、C 均为 row-major；bias（若非空）已偏移到当前 group，故 bias[m] 作用于输出第 m 行。
__global__ void gemm_tiled_kernel(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    const int M,
    const int N,
    const int K) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;

    float acc = 0.0f;
    const int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTile + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        const int b_row = t * kTile + threadIdx.y;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int e = 0; e < kTile; ++e) {
            acc += tile_a[threadIdx.y][e] * tile_b[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (bias != nullptr) {
            acc += bias[row];
        }
        C[row * N + col] = acc;
    }
}

}  // namespace

cudaError_t conv2d_nchw_im2col_gemm(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int n,
    const int c_in,
    const int h_in,
    const int w_in,
    const int c_out,
    const int k_h,
    const int k_w,
    const int h_out,
    const int w_out,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    cudaStream_t stream) {
    if (conv2d_internal::conv2d_args_invalid(
            input, weight, output, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
            stride_h, stride_w, dilation_h, dilation_w, groups)) {
        return cudaErrorInvalidValue;
    }

    const int in_channels_per_group = c_in / groups;
    const int out_channels_per_group = c_out / groups;
    const int M = out_channels_per_group;
    const int N = h_out * w_out;
    const int K = in_channels_per_group * k_h * k_w;

    // 单个 (batch, group) 子 GEMM 用的临时列矩阵。
    float* col = nullptr;
    cudaError_t status =
        cudaMallocAsync(&col, sizeof(float) * static_cast<std::size_t>(K) * N, stream);
    if (status != cudaSuccess) {
        return status;
    }

    const dim3 im2col_block(16, 16);
    const dim3 im2col_grid((N + 15) / 16, (K + 15) / 16);
    const dim3 gemm_block(kTile, kTile);
    const dim3 gemm_grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);

    for (int batch = 0; batch < n; ++batch) {
        for (int group = 0; group < groups; ++group) {
            const int ic_begin = group * in_channels_per_group;
            const int oc_begin = group * out_channels_per_group;

            im2col_kernel<<<im2col_grid, im2col_block, 0, stream>>>(
                input, col, batch, ic_begin, c_in, h_in, w_in, in_channels_per_group,
                k_h, k_w, h_out, w_out, stride_h, stride_w, pad_h, pad_w,
                dilation_h, dilation_w, K, N);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                cudaFreeAsync(col, stream);
                return status;
            }

            // W 子块：[M, K]，从第 oc_begin 行开始（weight 是 row-major 的
            // [c_out, K]）。output 子块：当前 (batch, group) 对应的 [M, N]。
            const float* weight_group = weight + static_cast<std::size_t>(oc_begin) * K;
            const float* bias_group = bias == nullptr ? nullptr : bias + oc_begin;
            float* output_group =
                output + (static_cast<std::size_t>(batch) * c_out + oc_begin) * N;

            gemm_tiled_kernel<<<gemm_grid, gemm_block, 0, stream>>>(
                weight_group, col, bias_group, output_group, M, N, K);
            status = cudaGetLastError();
            if (status != cudaSuccess) {
                cudaFreeAsync(col, stream);
                return status;
            }
        }
    }

    return cudaFreeAsync(col, stream);
}

}  // namespace wgkernel::cuda
