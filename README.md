# wGKernel

`wGKernel` 是一个面向通用高性能算子工程师方向的学习与实践仓库，用于系统实现、优化并记录各类高性能算子。

这个仓库的目标不是简单收集算子代码，而是围绕工程能力持续积累作品，包括：

- 算子数学定义与输入输出语义
- 朴素实现与高性能实现
- 正确性验证
- benchmark 与性能对比
- 优化思路沉淀，例如 `tiling`、`vectorization`、`memory coalescing`、`shared memory`、`warp-level optimization`、`fusion`

## 项目目标

- 面向求职场景，系统实现高频高价值算子
- 形成可展示的高性能算子作品集
- 持续积累 CUDA / CPU / 量化 / 融合优化经验
- 为后续面试准备提供统一的代码与文档入口

## 当前范围

当前阶段先专注 NVIDIA CUDA 平台，后续再扩展到更多平台。

## 规划方向

当前仓库计划围绕以下算子大类展开：

1. `gemm / matmul`
2. `gemv`
3. `reduce`
4. `norm`
5. `embedding / indexing`
6. `activation / elementwise`
7. `attention`
8. `convolution`
9. `quantization`
10. `transpose / layout transform`
11. `fused ops`

## 实现原则

每个算子尽量按照统一方式推进：

1. 明确问题定义与接口形式
2. 实现 baseline 版本
3. 实现高性能版本
4. 完成正确性验证
5. 补充 benchmark、profiling 与优化记录

## 项目结构

```text
wGKernel/
├── CMakeLists.txt
├── cmake/
├── include/
├── cpu/
├── cuda/
├── docs/
├── tests/
├── benchmarks/
├── ncu/
└── README.md
```

各目录职责如下：

- `cmake/`：统一管理 CMake 模块、CUDA 配置与公共构建逻辑
- `include/`：公共头文件与对外接口声明
- `cpu/`：CPU 算子实现、优化指南与平台侧学习笔记
- `cuda/`：CUDA 算子实现与各类别学习笔记
- `docs/`：跨算子、跨平台的学习问题与专题笔记
- `tests/`：统一的正确性测试
- `benchmarks/`：统一的性能 benchmark 与对标策略
- `ncu/`：Nsight Compute 脚本、指标集合、报告与分析记录

## 正确性与性能对标

项目将“正确性验证”和“性能 benchmark”分开维护。

正确性测试默认优先对齐 `PyTorch`，性能 benchmark 则按算子类别分别对齐工业界常用实现：

| 类别 | 正确性参考 | 性能对标 |
| --- | --- | --- |
| `gemm / gemv` | `PyTorch matmul` | `cuBLAS / cuBLASLt` |
| `reduce` | `PyTorch` | `PyTorch` |
| `norm` | `PyTorch` | `PyTorch` |
| `embedding / indexing` | `PyTorch` | `PyTorch` |
| `activation / elementwise` | `PyTorch` | `PyTorch` |
| `attention` | `PyTorch reference attention / SDPA` | `FlashAttention` |
| `convolution` | `PyTorch conv` | `cuDNN` |
| `quantization` | `PyTorch / reference implementation` | `cuBLASLt / CUTLASS` |
| `transpose / layout transform` | `PyTorch contiguous / permute` | `effective bandwidth ceiling` |
| `fused ops` | `PyTorch 组合表达式` | `PyTorch / fused reference implementation` |

## 构建方式

项目使用 `CMake` 管理构建。

```bash
cmake -S . -B build
cmake --build build -j
```

常用选项：

- `-DWGKERNEL_BUILD_TESTS=ON/OFF`
- `-DWGKERNEL_BUILD_BENCHMARKS=ON/OFF`
- `-DWGKERNEL_ENABLE_CUDA=ON/OFF`
- `-DCMAKE_CUDA_ARCHITECTURES=native`

## 当前实现进度

> 状态说明：
> `Not Started`：尚未开始
> `In Progress`：正在实现
> `Done`：已完成并通过基本验证

| 类别 | 代表算子 | 当前状态 | 已实现 | 待实现 |
| --- | --- | --- | --- | --- |
| `gemm / matmul` | `sgemm`、`hgemm`、`batched gemm` | `Not Started` | - | `sgemm`、`hgemm`、`batched gemm` |
| `gemv` | `sgemv`、`batched gemv` | `Not Started` | - | `sgemv`、`batched gemv` |
| `reduce` | `sum`、`max`、`argmax`、`softmax`、`logsumexp` | `In Progress` | `sum`、`max`、`argmax` | `softmax`、`logsumexp` |
| `norm` | `layernorm`、`rmsnorm`、`groupnorm` | `Not Started` | - | `layernorm`、`rmsnorm`、`groupnorm` |
| `embedding / indexing` | `embedding`、`gather`、`scatter` | `Not Started` | - | `embedding`、`gather`、`scatter` |
| `activation / elementwise` | `relu`、`gelu`、`silu`、`bias+activation` | `Not Started` | - | `relu`、`gelu`、`silu`、`bias+activation` |
| `attention` | `qk matmul`、`masked softmax`、`attention`、`flash attention` | `Not Started` | - | `qk matmul`、`masked softmax`、`attention`、`flash attention` |
| `convolution` | `conv2d`、`depthwise conv`、`im2col + gemm` | `Not Started` | - | `conv2d`、`depthwise conv`、`im2col + gemm` |
| `quantization` | `quantize`、`dequantize`、`int8 gemm`、`fp8 ops` | `Not Started` | - | `quantize`、`dequantize`、`int8 gemm`、`fp8 ops` |
| `transpose / layout transform` | `transpose`、`permute`、`nchw <-> nhwc` | `Not Started` | - | `transpose`、`permute`、`nchw <-> nhwc` |
| `fused ops` | `bias+relu`、`bias+gelu`、`residual+norm` | `Not Started` | - | `bias+relu`、`bias+gelu`、`residual+norm` |

## 后续计划

- 优先完成 `gemm / gemv / reduce / transpose`
- 为每类算子补充最小可运行示例
- 增加统一的测试、benchmark 与 profiling 脚手架
- 逐步补充优化分析文档

## 维护方式

当某个算子完成后，更新上表中的：

- `当前状态`
- `已实现`
- `待实现`

后续也可以继续补充：

- 对应源码路径
- benchmark 结果
- 优化说明
