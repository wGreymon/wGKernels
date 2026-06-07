# CPU Optimization Guide

`cpu/` 目录用于记录 CPU 侧高性能算子的实现、优化方法与实验结论。

当前阶段先作为 CPU 优化指南的占位入口，后续会逐步补充具体算子实现、benchmark、profiling 与优化笔记。

## 目标

- 系统掌握 CPU 高性能算子的常见优化手段
- 为 CUDA 算子提供 CPU baseline 和参考实现
- 积累 SIMD、多线程、cache 优化、NUMA 等工程经验
- 形成可用于面试复盘的 CPU 优化知识库

## 优化方向

后续计划围绕以下主题展开：

| 方向 | 关键词 | 适用算子 |
| --- | --- | --- |
| SIMD 向量化 | `SSE`、`AVX2`、`AVX-512`、intrinsics | `activation`、`reduce`、`norm`、`gemv` |
| Cache 优化 | blocking、tiling、cache locality | `gemm`、`conv`、`attention` |
| 多线程并行 | OpenMP、thread pool、work partition | `gemm`、`reduce`、`norm` |
| 内存布局 | contiguous、alignment、packing | `gemm`、`conv`、`embedding` |
| 循环优化 | unroll、software pipeline、branch reduction | elementwise、reduce |
| 预取 | software prefetch、streaming load/store | `embedding`、`gemv` |
| NUMA 优化 | thread affinity、first touch | 大规模矩阵计算 |
| 算子融合 | fusion、减少中间结果读写 | `bias+activation`、`residual+norm` |

## 推荐学习顺序

1. `activation / elementwise`
2. `reduce`
3. `norm`
4. `gemv`
5. `gemm`
6. `convolution`
7. `attention`

## 每个算子的记录模板

后续每个 CPU 算子建议记录：

```text
Operator:
Input shape:
Dtype:
Baseline:
Optimized version:
Correctness reference:
Benchmark command:
Latency:
Throughput:
Optimization notes:
Next step:
```

## Profiling 工具

后续可逐步引入：

- `perf`
- `vtune`
- `likwid`
- `valgrind/cachegrind`
- `google benchmark`

## 当前状态

- `Not Started`
