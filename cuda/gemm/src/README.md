# GEMM Source Layout

`cuda/gemm/src/` 用于放置 GEMM / Matmul 的不同实现版本。

建议文件组织：

- `sgemm.cu`：CUDA Core 路线的 FP32 GEMM。
- `hgemm_wmma.cu`：基于 WMMA API 的 Tensor Core HGEMM。
- `hgemm_mma.cu`：基于 MMA PTX 的 Tensor Core HGEMM。
- `hgemm_cutlass.cu`：基于 CUTLASS 的 HGEMM 示例。
- `gemm_bias_activation_cutlass.cu`：CUTLASS epilogue fusion 示例。

原则：

- 按具体算子组织源码，不单独建立 `tensor_core/` 根目录。
- Tensor Core 作为 GEMM 的实现技术路线体现在文件名和 notes 中。
- 每个文件内可以包含多个逐步优化版本，保持 kernel 代码聚焦、可读。
