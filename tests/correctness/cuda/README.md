# CUDA Correctness Strategy

当前阶段的 CUDA 正确性验证原则如下：

| 类别 | 正确性参考 |
| --- | --- |
| `gemm / gemv` | `PyTorch matmul` |
| `reduce` | `PyTorch` |
| `norm` | `PyTorch` |
| `embedding / indexing` | `PyTorch` |
| `activation / elementwise` | `PyTorch` |
| `attention` | `PyTorch reference attention / SDPA` |
| `convolution` | `PyTorch conv` |
| `quantization` | `PyTorch / reference implementation` |
| `transpose / layout transform` | `PyTorch contiguous / permute` |
| `fused ops` | `PyTorch 组合表达式` |

建议后续每个算子至少覆盖：

- 常见 shape
- 边界 shape
- 多种 dtype
- 连续与非连续输入
- 与参考实现的误差阈值校验
