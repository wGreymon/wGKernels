# CUTLASS Notes

CUTLASS 是 NVIDIA 开源的 CUDA C++ 模板库，用于实现高性能 GEMM、Convolution 和 Tensor Core 算子。

本文件用于记录 CUTLASS 学习路线和使用经验。真正引入 CUTLASS 源码或子模块时，建议放在：

```text
third_party/cutlass/
```

基于 CUTLASS 的算子示例仍然放在对应算子目录下，例如：

```text
cuda/gemm/src/hgemm_cutlass.cu
cuda/convolution/src/conv2d_cutlass.cu
```

## Learning Path

1. 先手写 CUDA Core GEMM，理解 tiling 和 memory hierarchy
2. 使用 WMMA 写 Tensor Core GEMM
3. 学习 MMA PTX 和 shared memory pipeline
4. 阅读 CUTLASS GEMM example
5. 理解 CUTLASS 的 tile 层级
6. 学习 epilogue fusion，例如 `gemm + bias + activation`
7. 进一步学习 CUTLASS 3.x / CuTe

## Key Topics

- threadblock tile
- warp tile
- instruction tile
- global memory iterator
- shared memory iterator
- mainloop pipeline
- epilogue
- layout / swizzle
- CuTe tensor layout abstraction

## Implementation Targets

| Operator | File | Status |
| --- | --- | --- |
| CUTLASS GEMM example | `cuda/gemm/src/hgemm_cutlass.cu` | `Not Started` |
| CUTLASS epilogue fusion | `cuda/gemm/src/gemm_bias_activation_cutlass.cu` | `Not Started` |
| CUTLASS convolution example | `cuda/convolution/src/conv2d_cutlass.cu` | `Not Started` |

## Notes

- CUTLASS 不是替代手写 kernel 的捷径，而是学习工业级 GEMM/Conv 实现结构的重要参考。
- 当前阶段先保留学习笔记，不急于引入完整源码。
- 当需要依赖 CUTLASS 编译示例时，再新增 `third_party/cutlass/` 和对应 CMake 配置。
