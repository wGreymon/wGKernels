# CPU Baseline Guide

`cpu/` 目录用于实现 CPU 侧朴素算子、记录参考行为与实验结论。

CPU 后端只保留朴素实现，用作 CUDA、CUTLASS、SIMD 等后端的正确性基线。

指令集加速实现单独放在 `simd/`，不混在 `cpu/` 目录中。

## 目标

- 系统掌握 CPU 高性能算子的常见优化手段
- 为 CUDA / CUTLASS 算子提供 CPU baseline 和参考实现
- 为后续 SIMD、多线程、cache 优化提供可信 reference
- 形成可用于面试复盘的 CPU 优化知识库

## 关注方向

后续计划围绕以下主题展开：

| 方向 | 关键词 | 适用算子 |
| --- | --- | --- |
| 标量 baseline | scalar loop、clear semantics | 所有算子 |
| 数值行为 | tie-break、rounding、edge cases | `reduce`、`norm`、`softmax` |
| 参考实现 | correctness reference、debug oracle | 所有算子 |

## 推荐学习顺序

1. `activation`
2. `elementwise`
3. `reduce`
4. `norm`
5. `gemv`
6. `gemm`
7. `convolution`
8. `attention`

## 每个算子的记录模板

后续每个 CPU 算子建议记录：

```text
Operator:
Input shape:
Dtype:
CPU baseline:
Correctness reference:
Benchmark command:
Latency:
Throughput:
Behavior notes:
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

- 已建立算子目录骨架。
- 下一步建议继续补齐 `cpu/reduce` 的更多 baseline 场景，并按同样方式扩展其他算子。
