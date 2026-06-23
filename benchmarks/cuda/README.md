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
- 算法操作吞吐 `GOPS`

说明：

- `sum` 的 reduce benchmark 会额外输出 `GFLOP/s`，操作量按 `numel - 1` 次加法估算。
- `max` 和 `argmax` 的操作量主要是比较操作，因此输出 `GOPS`，不强行记作 `GFLOP/s`。

当前文件：

- `reduce_benchmark.cu`：wGKernel reduce benchmark / NCU driver。
- `bench_reduce_torch.py`：PyTorch reduce 参考 benchmark。

后续计划：

- `conv2d_benchmark.cu`：wGKernel conv2d benchmark / NCU driver。
