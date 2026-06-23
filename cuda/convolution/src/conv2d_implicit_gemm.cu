// 算法 4：隐式 GEMM（implicit GEMM）。
//
// 概念上与算法 2 完全一致：对每个 (batch, group) 计算
//   C[M, N] = W[M, K] * Col[K, N]
// M/N/K 含义相同。区别在于 Col 永远不写回 global memory。当 GEMM 需要某个
// shared tile 里的 Col[k][p] 时，本 kernel 现场把 k -> (icg, kh, kw)、
// p -> (oh, ow) 解码，直接从 input gather 对应元素（padding 返回 0）。这样既省掉
// 了 im2col pass，也避免了 K 倍的显存膨胀，所以 cuDNN/CUTLASS 都偏好这条路。
//
// Grid：z = batch * groups，(x, y) 切分子 GEMM 的 (N, M) 输出。

#include "cuda/convolution.hpp"

#include <cuda_runtime.h>

#include "conv2d_internal.cuh"

namespace wgkernel::cuda {
namespace {

constexpr int kTile = 16;  // GEMM tile 边长

__global__ void conv2d_nchw_implicit_gemm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
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
    const int in_channels_per_group,
    const int out_channels_per_group,
    const int M,
    const int N,
    const int K) {
    __shared__ float tile_w[kTile][kTile];   // A tile：weight 行
    __shared__ float tile_col[kTile][kTile]; // B tile：gather 得到的输入列

    const int bg = blockIdx.z;
    const int batch = bg / groups;
    const int group = bg % groups;
    const int oc_begin = group * out_channels_per_group;
    const int ic_begin = group * in_channels_per_group;

    const int m = blockIdx.y * kTile + threadIdx.y;  // 局部输出通道（行）
    const int p = blockIdx.x * kTile + threadIdx.x;  // 输出空间位置（列）

    const int oc = oc_begin + m;
    const int ow = p % w_out;
    const int oh = p / w_out;

    float acc = 0.0f;
    const int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        // 载入 A tile：W[m, t*kTile + tx]。当前 group 的 weight 子块在内存中是
        // 连续 row-major 的 [out_channels_per_group, K]。
        const int a_k = t * kTile + threadIdx.x;
        tile_w[threadIdx.y][threadIdx.x] =
            (m < M && a_k < K) ? weight[oc * K + a_k] : 0.0f;

        // 通过从 input gather Col[t*kTile + ty, p] 来载入 B tile。
        const int b_k = t * kTile + threadIdx.y;
        float col_value = 0.0f;
        if (b_k < K && p < N) {
            const int kw = b_k % k_w;
            int tmp = b_k / k_w;
            const int kh = tmp % k_h;
            const int icg = tmp / k_h;
            const int ic = ic_begin + icg;
            const int ih = oh * stride_h - pad_h + kh * dilation_h;
            const int iw = ow * stride_w - pad_w + kw * dilation_w;
            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                col_value = input[((batch * c_in + ic) * h_in + ih) * w_in + iw];
            }
        }
        tile_col[threadIdx.y][threadIdx.x] = col_value;

        __syncthreads();

        for (int e = 0; e < kTile; ++e) {
            acc += tile_w[threadIdx.y][e] * tile_col[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (m < M && p < N) {
        if (bias != nullptr) {
            acc += bias[oc];
        }
        output[((batch * c_out + oc) * h_out + oh) * w_out + ow] = acc;
    }
}

}  // namespace

cudaError_t conv2d_nchw_implicit_gemm(
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

    const dim3 block(kTile, kTile);
    const dim3 grid(
        (N + kTile - 1) / kTile,
        (M + kTile - 1) / kTile,
        static_cast<unsigned>(n * groups));
    conv2d_nchw_implicit_gemm_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
        in_channels_per_group, out_channels_per_group, M, N, K);
    return cudaGetLastError();
}

}  // namespace wgkernel::cuda
