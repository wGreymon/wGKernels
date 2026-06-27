# SIMD Backend

`simd/` 存放 CPU 指令集加速版本。

定位：

- `cpu/` 只保留朴素 CPU baseline。
- `simd/` 负责 SSE、AVX2、AVX-512、NEON 等指令级优化实现。
- SIMD 实现必须对齐 `cpu/` baseline 的数值行为。

当前阶段先建立目录骨架，具体算子逐步补齐。
