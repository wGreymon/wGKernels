# Tensor Core Notes

Tensor Core 是矩阵乘累加专用计算单元，适合 GEMM、Convolution、Attention 中的大规模矩阵乘。

本文件用于记录 Tensor Core 编程相关知识，不作为独立算子目录。具体实现应放在对应算子目录下，例如：

```text
cuda/gemm/src/hgemm_wmma.cu
cuda/gemm/src/hgemm_mma.cu
cuda/convolution/src/implicit_gemm_tensorcore.cu
cuda/attention/src/flash_attention_tensorcore.cu
```

## Learning Path

1. 使用 WMMA 写最小可运行 `hgemm`
2. 理解 fragment、warp tile、matrix layout 和 accumulator
3. 对比 CUDA Core `hgemm` 与 Tensor Core `hgemm`
4. 学习 `ldmatrix`、`mma.sync` 和 PTX MMA
5. 实现基于 shared memory pipeline 的 Tensor Core GEMM
6. 使用 Nsight Compute 分析 Tensor Core utilization
7. 阅读 CUTLASS 中对应层级的抽象

## Key Topics

- WMMA API
- MMA PTX
- `mma.sync`
- `ldmatrix`
- warp-level matrix tile
- shared memory layout / swizzle
- double buffering / ping-pong buffer
- Tensor Core utilization
- mixed precision accumulation

## Implementation Targets

| Operator | File | Status |
| --- | --- | --- |
| WMMA HGEMM | `cuda/gemm/src/hgemm_wmma.cu` | `Not Started` |
| MMA HGEMM | `cuda/gemm/src/hgemm_mma.cu` | `Not Started` |
| Tensor Core attention matmul | `cuda/attention/src/flash_attention_tensorcore.cu` | `Not Started` |

## Notes

- Tensor Core 是实现方式，不是单独的算子类别。
- 普通 CUDA C++ `acc += a * b` 默认不会自动变成 Tensor Core 指令。
- 使用 Tensor Core 通常需要 WMMA、MMA PTX、CUTLASS、cuBLAS、cuDNN 或 FlashAttention 这类路径。
