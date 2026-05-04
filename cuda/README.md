# CUDA Operators

`cuda/` 目录用于存放各类 CUDA 高性能算子的实现与优化记录。

当前阶段先专注 NVIDIA CUDA 平台。后续如需支持更多平台，统一测试目录将放在项目根目录下的 `tests/` 中。

当前按算子大类划分为：

- `gemm`
- `gemv`
- `reduce`
- `norm`
- `embedding_indexing`
- `activation_elementwise`
- `attention`
- `convolution`
- `quantization`
- `transpose_layout_transform`
- `fused_ops`

每个类别目录下提供一个 `notes.md`，用于记录：

- 代表算子
- 实现计划
- 优化思路
- benchmark 与 profiling 结论
