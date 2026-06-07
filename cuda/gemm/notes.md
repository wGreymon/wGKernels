# GEMM Notes

## Scope

- `sgemm`
- `hgemm`
- `batched gemm`
- `tensor core gemm`

## Status

- `Not Started`

## Notes

- Record kernel design, tiling strategy, memory hierarchy usage, and performance comparisons here.

## Source Layout

- `src/sgemm.cu`: CUDA Core FP32 GEMM.
- `src/hgemm_wmma.cu`: Tensor Core GEMM through WMMA.
- `src/hgemm_mma.cu`: Tensor Core GEMM through MMA PTX.
- `src/hgemm_cutlass.cu`: CUTLASS-based GEMM example.

Tensor Core is treated as an implementation path under GEMM instead of a standalone operator category.
