# CUDA Tests

每个算子在本目录下各占一个 `test_<算子名>/` 目录，内部再分两个子目录：

- `scripts/`：Python 测试脚本。脚本直接构造 CUDA tensor，通过统一的 `wgkernel_cuda`
  pybind11 扩展模块调用 CUDA 算子，并与 PyTorch 对拍。
- `profiling/`：该算子的三层性能分析结果。`<算子>_performance_analysis.md` 为分析报告，
  对应的 `*.ncu-rep` 原始数据也直接放在该目录下（体积较大，已 gitignore）。

新增算子 = 复制一个 `test_<算子名>/` 目录，再在 `tests/cuda/CMakeLists.txt` 加一行
`add_subdirectory`。

## 正确性验证对标

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
