# CUDA Benchmark Strategy

当前阶段的 CUDA benchmark 对标策略如下：

| 类别 | 性能对标 |
| --- | --- |
| `gemm / gemv` | `cuBLAS / cuBLASLt` |
| `reduce` | `PyTorch` |
| `norm` | `PyTorch` |
| `embedding / indexing` | `PyTorch` |
| `activation / elementwise` | `PyTorch` |
| `attention` | `FlashAttention` |
| `convolution` | `cuDNN` |
| `quantization` | `cuBLASLt / CUTLASS` |
| `transpose / layout transform` | `effective bandwidth ceiling` |
| `fused ops` | `PyTorch / fused reference implementation` |

建议 benchmark 输出至少包含：

- GPU 型号
- CUDA / driver 版本
- shape
- dtype
- warmup 次数
- repeat 次数
- latency
- speedup
- 计算密集型算子的 `TFLOPS`
- 访存密集型算子的 `GB/s`
