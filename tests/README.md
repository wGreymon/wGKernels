# Tests

`tests/` 用于存放统一的正确性测试。

当前项目先专注 NVIDIA CUDA 平台，但测试目录仍然按平台无关的方式组织，便于后续扩展到更多后端。

## 组织方式

- 第一层按平台分目录：`tests/cuda/`（后续可扩展 `tests/metaX/` 等）。
- 平台目录下每个算子各自一个 `test_<算子名>/` 目录，自包含：
  - `scripts/test_<算子>_vs_pytorch.py`：Python 端直接生成输入、调用统一 Python 扩展模块、和 PyTorch 对拍。
- 所有 CUDA 算子统一编译进 `wgkernel_cuda` Python 扩展模块；不同算子的行为差异由各自
  Python 测试脚本负责。
- `profiling/`：可选目录，用于存放该算子的三层性能分析报告和对应的 profiler 原始数据。
- 新增算子 = 复制一个 `test_<算子名>/` 目录，再在 `tests/cuda/CMakeLists.txt` 里加一行 `add_subdirectory`。

当前约定是：算子相关的性能分析跟随对应测试目录维护，例如
`tests/cuda/test_reduce/profiling/`。

示例：

```bash
PYTHONPATH=build/python python3 tests/cuda/test_reduce/scripts/test_reduce_vs_pytorch.py
PYTHONPATH=build/python python3 tests/cuda/test_conv2d/scripts/test_conv2d_vs_pytorch.py
```
