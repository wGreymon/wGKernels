# CUDA 算子开发计划

`cuda/` 目录用于存放各类 CUDA 高性能算子的实现、测试、benchmark 和 profiling 记录。

当前阶段目标：

- 先覆盖常见深度学习和 YOLOX 推理算子。
- 每个算子先实现 naive baseline，保证语义清楚、可测试。
- 后续逐个进行正确性对齐、benchmark 和性能优化。
- 优先对齐 NVIDIA CUDA 平台，后续再扩展到其它平台。

## 状态说明

| 状态 | 含义 |
| --- | --- |
| `Not Started` | 尚未开始 |
| `Planned` | 已规划目录或文档，但还没有实现 |
| `Naive Implemented` | 已有 naive CUDA 实现，尚未充分测试或优化 |
| `Correctness Ready` | 已有正确性测试并对齐参考实现 |
| `Benchmark Ready` | 已有 benchmark 和性能对标 |
| `Optimizing` | 正在进行性能优化 |
| `Done` | 当前阶段完成 |

## 总览

| 类别 | 目录 | 当前状态 | 正确性参考 | 性能对标 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| `reduce` | `cuda/reduce/` | `Benchmark Ready` | PyTorch | PyTorch | 补 row-wise reduce、softmax、logsumexp |
| `activation / elementwise` | `cuda/activation_elementwise/` | `Naive Implemented` | PyTorch | PyTorch | 补 correctness test 和 benchmark |
| `convolution` | `cuda/convolution/` | `Naive Implemented` | PyTorch `conv2d` | cuDNN | 优先补 Conv2d correctness test |
| `norm` | `cuda/norm/` | `Naive Implemented` | PyTorch | PyTorch | 补 BatchNorm2d inference correctness test |
| `pooling` | `cuda/pooling/` | `Naive Implemented` | PyTorch | PyTorch / cuDNN | 补 MaxPool2d correctness test |
| `resize` | `cuda/resize/` | `Naive Implemented` | PyTorch | PyTorch | 补 nearest/bilinear correctness test |
| `transpose / layout transform` | `cuda/transpose_layout_transform/` | `Naive Implemented` | PyTorch | effective bandwidth | 补 concat、permute correctness test |
| `embedding / indexing` | `cuda/embedding_indexing/` | `Naive Implemented` | PyTorch | PyTorch | 补 slice、gather、topk、sort correctness test |
| `gemm / matmul` | `cuda/gemm/` | `Planned` | PyTorch / cuBLAS | cuBLAS / cuBLASLt | 实现 CUDA Core SGEMM baseline |
| `gemv` | `cuda/gemv/` | `Planned` | PyTorch | cuBLAS | 实现 SGEMV baseline |
| `attention` | `cuda/attention/` | `Planned` | PyTorch SDPA | FlashAttention | 先明确 standard / linear / flash-like 接口 |
| `convolution depthwise` | `cuda/convolution/` | `Planned` | PyTorch | cuDNN | 在 Conv2d correctness 后实现 depthwise case benchmark |
| `quantization` | `cuda/quantization/` | `Planned` | PyTorch / reference | cuBLASLt / CUTLASS | 实现 quantize / dequantize baseline |
| `fused ops` | `cuda/fused_ops/` | `Planned` | PyTorch 组合表达式 | fused reference | 暂缓，先不考虑融合 |

## YOLOX 相关算子落点

| YOLOX 算子 | CUDA 目录 | 当前实现 | 备注 |
| --- | --- | --- | --- |
| `Conv2d` | `cuda/convolution/` | `conv2d_nchw` | 支持 bias、stride、padding、dilation、groups |
| `BatchNorm2d` | `cuda/norm/` | `batchnorm2d_inference_nchw` | 推理态 per-channel affine |
| `SiLU / Swish` | `cuda/activation_elementwise/` | `silu` | unary elementwise |
| `Sigmoid` | `cuda/activation_elementwise/` | `sigmoid` | unary elementwise |
| `Exp` | `cuda/activation_elementwise/` | `exp` | bbox decode 相关 |
| `Add / Sub / Mul` | `cuda/activation_elementwise/` | `add / sub / mul` | binary elementwise |
| `MaxPool2d` | `cuda/pooling/` | `maxpool2d_nchw` | SPP 模块 |
| `Upsample` | `cuda/resize/` | `upsample_nearest2d_nchw`、`upsample_bilinear2d_nchw` | FPN 自顶向下路径 |
| `Concat` | `cuda/transpose_layout_transform/` | `concat_nchw_axis1` | 当前支持 NCHW channel 维拼接 |
| `Reshape / View` | `cuda/transpose_layout_transform/` | `copy_1d` | view 通常是 metadata，不一定需要 kernel |
| `Permute` | `cuda/transpose_layout_transform/` | `permute_nchw_to_nhwc`、`permute_nhwc_to_nchw` | 先覆盖 NCHW/NHWC |
| `Slice / Strided Slice` | `cuda/embedding_indexing/` | `slice_1d` | 当前是 1D baseline |
| `Gather` | `cuda/embedding_indexing/` | `gather_1d` | 当前是 1D baseline |
| `TopK / Sort` | `cuda/embedding_indexing/` | `topk_1d`、`sort_1d` | 当前是单线程 naive baseline |

## 短期优先级

1. 为 YOLOX naive 算子补正确性测试，对齐 PyTorch。
2. 优先测试 `Conv2d -> BatchNorm2d -> SiLU -> MaxPool2d -> Upsample -> Concat`。
3. 为每个算子补最小 benchmark，输出 `latency`、`GB/s`、`GOPS/GFLOP/s`。
4. 优先优化 `Conv2d`，因为它是 YOLOX 主要性能热点。
5. 再优化 `MaxPool2d`、`Upsample`、`Concat` 等视觉模型常见 memory-bound 算子。

## 正确性测试计划

| 测试文件 | 覆盖算子 | 状态 |
| --- | --- | --- |
| `tests/cuda/test_reduce/test_reduce_vs_pytorch.py` | `sum`、`max`、`argmax` | `Done` |
| `tests/cuda/test_conv2d/test_conv2d_vs_pytorch.py` | Conv2d | `Done` |
| `tests/cuda/test_yolox_ops/test_yolox_ops_vs_pytorch.py` | BatchNorm2d、SiLU、MaxPool2d、Upsample、Concat | `Planned` |
| `tests/cuda/test_activation/test_activation_vs_pytorch.py` | SiLU、Sigmoid、Exp、Add、Sub、Mul | `Planned` |
| `tests/cuda/test_transform/test_transform_vs_pytorch.py` | Concat、Permute、Slice、Gather | `Planned` |

## Benchmark 计划

| benchmark | 覆盖算子 | 对标 |
| --- | --- | --- |
| `benchmarks/cuda/reduce_benchmark.cu` | reduce | PyTorch |
| `benchmarks/cuda/yolox_ops_benchmark.cu` | Conv2d、BN、SiLU、MaxPool、Upsample、Concat | PyTorch / cuDNN |
| `benchmarks/cuda/conv2d_benchmark.cu` | Conv2d | cuDNN |
| `benchmarks/cuda/activation_benchmark.cu` | unary/binary elementwise | PyTorch |
| `benchmarks/cuda/transform_benchmark.cu` | concat、permute、copy | effective bandwidth |

## 优化路线

### Conv2d

1. naive direct convolution
2. 支持常见 YOLOX shape correctness
3. benchmark 对齐 PyTorch / cuDNN
4. im2col + GEMM
5. direct tiled convolution
6. implicit GEMM
7. Tensor Core / CUTLASS convolution

### Elementwise / Activation

1. one thread per element baseline
2. vectorized load/store
3. fast math / approximation
4. memory bandwidth benchmark
5. 后续再考虑 fusion

### Pooling / Resize / Transform

1. naive baseline
2. coalesced read/write
3. shared memory 或 tile 优化
4. benchmark effective bandwidth
5. 特化 YOLOX 常见 shape

## 记录原则

- 算子目录不强制保留 `notes.md`，只有需要记录设计、实验或总结时再创建。
- 本文件维护 CUDA 算子开发计划和状态总览。
- 状态变更后同步更新本文件。
- benchmark 结果和 NCU 分析可以放到按需创建的算子笔记或 `ncu/notes/`。
