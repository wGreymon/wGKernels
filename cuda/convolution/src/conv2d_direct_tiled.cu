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

#include "cuda/convolution.hpp"

#include <cuda_runtime.h>

#include <cstddef>

#include "conv2d_internal.cuh"

namespace wgkernel::cuda {
namespace {

constexpr int kTile = 16;  // 输出空间 tile 的边长

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

}  // namespace

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
    if (conv2d_internal::conv2d_args_invalid(
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

}  // namespace wgkernel::cuda
