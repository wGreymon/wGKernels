# Tests

`tests/` 用于存放统一的正确性测试。

当前测试脚本对齐 `wGInfer` 的风格，通过参数指定算子的运行设备。

## 测试原则

测试脚本默认承担正确性验证职责：脚本运行时会构造输入、调用当前后端算子、与 PyTorch
或参考表达式对拍。如果结果不满足误差阈值，脚本必须直接抛出 `AssertionError` 并失败。

每个算子的 case 设计至少覆盖三类规模：

- `tiny/small`：边界行为和小输入，例如单元素、非 2 的幂、非方形 shape。
- `medium`：常规工作负载，便于快速回归。
- `large/throughput_like`：较大输入规模，用于让 CPU baseline、SIMD、CUDA、CUTLASS、MetaX
  之间的加速比更直观。

大 shape 仍然属于正确性测试的一部分，但不要把默认测试设计成压力测试。完整多规模性能曲线
和稳定 latency 测量应放到 `benchmarks/` 或对应算子的 `profiling/` 文档中。

## 组织方式

- 第一层按算子分目录，例如 `tests/test_reduce/`、`tests/test_conv2d/`。
- 每个算子目录自包含：
  - `scripts/test_<算子>.py`：Python 端直接生成输入、调用统一 Python 扩展模块、和 PyTorch 对拍。
- 各后端分别编译成独立 pybind 模块，例如 `wgkernel_cpu`、`wgkernel_simd`、
  `wgkernel_cuda`、`wgkernel_cutlass`、`wgkernel_metax`。
- 测试脚本根据 `--device` 动态加载 `wgkernel_<device>`，不同后端需要暴露一致的算子 API。
- 脚本使用 `--device` 指定运行设备，取值为 `cpu`、`simd`、`cuda`、`cutlass`、`metax`。
- `profiling/`：可选目录，用于存放该算子的三层性能分析报告和对应的 profiler 原始数据。
- 新增算子 = 复制一个 `test_<算子名>/` 目录，并在脚本里接入对应平台函数。

当前约定是：算子相关的性能分析跟随对应测试目录维护，例如
`tests/test_reduce/profiling/`。

示例：

```bash
PYTHONPATH=build/python python3 tests/test_reduce/scripts/test_reduce.py --device cuda
PYTHONPATH=build/python python3 tests/test_conv2d/scripts/test_conv2d.py --device cuda
PYTHONPATH=build/python python3 tests/test_activation/scripts/test_activation.py --device cpu --op-name all
PYTHONPATH=build/python python3 tests/test_activation/scripts/test_activation.py --device cpu --op-name silu
PYTHONPATH=build/python python3 tests/test_attention/scripts/test_attention.py --device cpu --op-name self_attention
```

当前已存在并逐步完善的 pybind 模块包括 `wgkernel_cpu` 和 `wgkernel_cuda`。`simd`、
`cutlass`、`metax` 会在对应模块编译完成后通过同一套测试脚本运行。
