#include "cuda/convolution.hpp"

#include <cuda_runtime.h>

#include <cstddef>

// 本文件用四种方式实现 conv2d，从最朴素到最接近 GEMM 的写法，方便对照学习：
//
//   1. conv2d_nchw                  - 朴素直接卷积，一线程算一个输出元素
//   2. conv2d_nchw_im2col_gemm      - 把输入展开成列矩阵，再做 tiled GEMM
//   3. conv2d_nchw_direct_tiled     - 输出分块，并把输入 halo 暂存到 shared memory
//   4. conv2d_nchw_implicit_gemm    - GEMM 分块，但列矩阵不落地，按需现算
//
// 四个版本共用相同的 NCHW 布局，语义与算法 1 一致，算法 1 同时充当正确性参考。

namespace wgkernel::cuda {
namespace {

constexpr int kBlockSize = 256;  // elementwise kernel 每个 block 的线程数
constexpr int kTile = 16;        // GEMM tile / 输出空间 tile 的边长

// 公共参数校验。参数非法时返回 true。
bool conv2d_args_invalid(
    const float* input,
    const float* weight,
    const float* output,
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
    const int dilation_h,
    const int dilation_w,
    const int groups) {
    return input == nullptr || weight == nullptr || output == nullptr || n <= 0 || c_in <= 0 ||
        h_in <= 0 || w_in <= 0 || c_out <= 0 || k_h <= 0 || k_w <= 0 || h_out <= 0 || w_out <= 0 ||
        stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 || groups <= 0 ||
        c_in % groups != 0 || c_out % groups != 0;
}

// ============================================================================
// 算法 1：朴素直接卷积。
//
// 一个线程负责一个输出元素 output[n][oc][oh][ow]，自己走完整个
// (in_channel, k_h, k_w) 的累加。简单且正确，但每个线程都从 global memory
// 重复读取 weight 和 input，没有任何复用。
// ============================================================================
__global__ void conv2d_nchw_kernel(
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
    const int total) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    int tmp = linear;
    const int ow = tmp % w_out;
    tmp /= w_out;
    const int oh = tmp % h_out;
    tmp /= h_out;
    const int oc = tmp % c_out;
    const int batch = tmp / c_out;

    const int out_channels_per_group = c_out / groups;
    const int in_channels_per_group = c_in / groups;
    const int group = oc / out_channels_per_group;
    const int ic_begin = group * in_channels_per_group;

    float acc = bias == nullptr ? 0.0f : bias[oc];

    for (int icg = 0; icg < in_channels_per_group; ++icg) {
        const int ic = ic_begin + icg;
        for (int kh = 0; kh < k_h; ++kh) {
            const int ih = oh * stride_h - pad_h + kh * dilation_h;
            if (ih < 0 || ih >= h_in) {
                continue;
            }
            for (int kw = 0; kw < k_w; ++kw) {
                const int iw = ow * stride_w - pad_w + kw * dilation_w;
                if (iw < 0 || iw >= w_in) {
                    continue;
                }
                const int input_index = ((batch * c_in + ic) * h_in + ih) * w_in + iw;
                const int weight_index = ((oc * in_channels_per_group + icg) * k_h + kh) * k_w + kw;
                acc += input[input_index] * weight[weight_index];
            }
        }
    }

    output[linear] = acc;
}

// ============================================================================
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
// ============================================================================

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

// ============================================================================
// 算法 3：带 shared memory 输入 halo 的直接分块卷积。
//
// 一个 block 为某个 (batch, oc) 计算一个 kTile x kTile 的输出 tile。对 group 内
// 的每个输入通道，先协作地把整个 tile 要读的输入区域（tile 加上其 halo 边缘）
// 一次性载入 shared memory，然后每个线程从 shared memory 取自己的 k_h x k_w
// 窗口做累加。这样就把算法 1 中重复的 global 读变成了 shared memory 复用。
//
// halo 大小取决于 stride/dilation/kernel，所以 shared memory 用动态分配：
//   sh_h = (kTile-1)*stride_h + (k_h-1)*dilation_h + 1
//   sh_w = (kTile-1)*stride_w + (k_w-1)*dilation_w + 1
// ============================================================================
__global__ void conv2d_nchw_direct_tiled_kernel(
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
    const int in_channels_per_group,
    const int out_channels_per_group,
    const int tiles_per_row,
    const int sh_h,
    const int sh_w) {
    extern __shared__ float patch[];  // 单个通道的 sh_h * sh_w 输入 halo

    // 解码 block -> (batch, oc, 输出 tile 起点)。
    const int tile_id = blockIdx.x;
    const int tiles_per_col = (h_out + kTile - 1) / kTile;
    const int tiles_per_image = tiles_per_row * tiles_per_col;

    int tmp = tile_id;
    const int tile_index = tmp % tiles_per_image;
    tmp /= tiles_per_image;
    const int oc = tmp % c_out;
    const int batch = tmp / c_out;

    const int tile_row = tile_index / tiles_per_row;
    const int tile_col = tile_index % tiles_per_row;
    const int oh0 = tile_row * kTile;
    const int ow0 = tile_col * kTile;

    const int oh = oh0 + threadIdx.y;
    const int ow = ow0 + threadIdx.x;

    const int group = oc / out_channels_per_group;
    const int ic_begin = group * in_channels_per_group;

    // 当前输出 tile 覆盖的输入区域左上角坐标。
    const int base_ih = oh0 * stride_h - pad_h;
    const int base_iw = ow0 * stride_w - pad_w;

    float acc = bias == nullptr ? 0.0f : bias[oc];

    for (int icg = 0; icg < in_channels_per_group; ++icg) {
        const int ic = ic_begin + icg;

        // 协作地把当前通道的输入 halo 暂存到 shared memory。
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x; idx < sh_h * sh_w;
             idx += blockDim.x * blockDim.y) {
            const int sr = idx / sh_w;
            const int sc = idx % sh_w;
            const int ih = base_ih + sr;
            const int iw = base_iw + sc;
            float value = 0.0f;
            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                value = input[((batch * c_in + ic) * h_in + ih) * w_in + iw];
            }
            patch[idx] = value;
        }
        __syncthreads();

        // 直接从 shared memory 取出本线程的窗口做累加。
        if (oh < h_out && ow < w_out) {
            const int sr0 = threadIdx.y * stride_h;
            const int sc0 = threadIdx.x * stride_w;
            for (int kh = 0; kh < k_h; ++kh) {
                const int sr = sr0 + kh * dilation_h;
                for (int kw = 0; kw < k_w; ++kw) {
                    const int sc = sc0 + kw * dilation_w;
                    const int weight_index =
                        ((oc * in_channels_per_group + icg) * k_h + kh) * k_w + kw;
                    acc += patch[sr * sh_w + sc] * weight[weight_index];
                }
            }
        }
        __syncthreads();
    }

    if (oh < h_out && ow < w_out) {
        output[((batch * c_out + oc) * h_out + oh) * w_out + ow] = acc;
    }
}

// ============================================================================
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
// ============================================================================
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
cudaError_t conv2d_nchw(
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
    if (conv2d_args_invalid(
            input, weight, output, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
            stride_h, stride_w, dilation_h, dilation_w, groups)) {
        return cudaErrorInvalidValue;
    }

    const int total = n * c_out * h_out * w_out;
    const int blocks = (total + kBlockSize - 1) / kBlockSize;
    conv2d_nchw_kernel<<<blocks, kBlockSize, 0, stream>>>(
        input, weight, bias, output, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, total);
    return cudaGetLastError();
}

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
    if (conv2d_args_invalid(
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

cudaError_t conv2d_nchw_direct_tiled(
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
    if (conv2d_args_invalid(
            input, weight, output, n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
            stride_h, stride_w, dilation_h, dilation_w, groups)) {
        return cudaErrorInvalidValue;
    }

    const int in_channels_per_group = c_in / groups;
    const int out_channels_per_group = c_out / groups;

    const int tiles_per_row = (w_out + kTile - 1) / kTile;
    const int tiles_per_col = (h_out + kTile - 1) / kTile;
    const int tiles_per_image = tiles_per_row * tiles_per_col;
    const int blocks = n * c_out * tiles_per_image;

    // tile 沿各轴需要读取的 halo 范围。
    const int sh_h = (kTile - 1) * stride_h + (k_h - 1) * dilation_h + 1;
    const int sh_w = (kTile - 1) * stride_w + (k_w - 1) * dilation_w + 1;
    const std::size_t shared_bytes = sizeof(float) * static_cast<std::size_t>(sh_h) * sh_w;

    const dim3 block(kTile, kTile);
    conv2d_nchw_direct_tiled_kernel<<<blocks, block, shared_bytes, stream>>>(
        input, weight, bias, output, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        in_channels_per_group, out_channels_per_group, tiles_per_row, sh_h, sh_w);
    return cudaGetLastError();
}

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
    if (conv2d_args_invalid(
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
