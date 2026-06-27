# wGKernel

`wGKernel` 是一个面向通用高性能算子工程师方向的学习与实践仓库，用于系统实现、优化并记录各类高性能算子。

这个仓库的目标开发达到工业级实现的kernel，包括：

- 算子数学定义与输入输出语义
- 朴素实现与高性能实现
- 正确性验证
- benchmark 与性能对比
- 优化思路沉淀，例如 `tiling`、`vectorization`、`memory coalescing`、`shared memory`、`warp-level optimization`、`fusion`

## 项目目标

- 面向求职场景，系统实现高频高价值算子
- 形成可展示的高性能算子作品集
- 持续积累 CUDA / CPU / 量化 / 融合优化经验
- 为后续面试准备提供统一的代码与文档入口

## 当前范围

当前阶段以 NVIDIA GPU 和 CPU baseline 为主，按同一算子、多种实现路线推进：

1. `cpu/`：CPU 朴素实现，作为正确性基线。
2. `simd/`：CPU 指令集加速实现。
3. `cuda/`：纯手写 CUDA kernel。
4. `cutlass/`：调用 CUTLASS 组件实现算子，作为工业级 GPU 实现参考。
5. `metaX/`：沐曦计算平台实现或占位，当前按平台目录保留结构。

## 规划方向

当前仓库计划围绕以下算子大类展开：

1. `gemm / matmul`
2. `gemv`
3. `reduce`
4. `norm`
5. `embedding / indexing`
6. `activation`
7. `elementwise`
8. `attention`
9. `convolution`
10. `quantization`
11. `layout`
12. `fused ops`

## 实现原则

每个算子尽量按照统一方式推进：

1. 明确问题定义与接口形式
2. 实现 baseline 版本
3. 实现高性能版本
4. 完成正确性验证
5. 补充 benchmark、profiling 与优化记录

## 项目结构

```text
wGKernel/
├── CMakeLists.txt
├── cmake/
├── include/
├── cpu/
├── simd/
├── cuda/
├── cutlass/
├── metaX/
├── docs/
├── tests/
├── benchmarks/
├── ncu/
└── README.md
```

各目录职责如下：

- `cmake/`：统一管理 CMake 模块、CUDA 配置与公共构建逻辑
- `include/`：公共头文件与对外接口声明
- `cpu/`：CPU 朴素 baseline
- `simd/`：CPU 指令集加速实现
- `cuda/`：纯手写 CUDA kernel 实现
- `cutlass/`：基于 CUTLASS 的 GPU 算子实现
- `metaX/`：沐曦计算平台实现或占位
- `docs/`：跨算子、跨平台的学习问题与专题笔记
- `tests/`：各后端正确性测试，Python 脚本直接运行
- `benchmarks/`：各后端性能 benchmark 与工业实现对标
- `ncu/`：Nsight Compute 脚本、指标集合、报告与分析记录

## 正确性与性能对标

项目将“正确性验证”和“性能 benchmark”分开维护。

正确性测试默认优先对齐 `PyTorch`，性能 benchmark 则按算子类别分别对齐工业界常用实现：

| 类别 | 正确性参考 | 性能对标 |
| --- | --- | --- |
| `gemm / gemv` | `PyTorch matmul` | `cuBLAS / cuBLASLt` |
| `reduce` | `PyTorch` | `PyTorch` |
| `norm` | `PyTorch` | `PyTorch` |
| `embedding / indexing` | `PyTorch` | `PyTorch` |
| `activation` | `PyTorch` | `PyTorch` |
| `elementwise` | `PyTorch` | `PyTorch` |
| `attention` | `PyTorch reference attention / SDPA` | `FlashAttention` |
| `convolution` | `PyTorch conv` | `cuDNN` |
| `quantization` | `PyTorch / reference implementation` | `cuBLASLt / CUTLASS` |
| `layout` | `PyTorch contiguous / permute` | `effective bandwidth ceiling` |
| `fused ops` | `PyTorch 组合表达式` | `PyTorch / fused reference implementation` |

## 算子实现性能表

本仓库的性能记录以 `CPU baseline` 为锚点。每个算子先记录 CPU 朴素实现的 latency，
之后再记录 CUDA、CUTLASS、PyTorch 等实现的 latency，便于横向计算加速比。

核心指标：

```text
speedup_vs_cpu = cpu_baseline_latency / implementation_latency
```

`speedup_vs_cpu > 1.0x` 表示该实现快于 CPU baseline。

完整 benchmark 命令、多规模曲线、NCU/perf 分析放在各算子的 `profiling/` 文档中；
README 只维护最适合横向比较的代表性结果。

当前 CPU baseline 数字来自 `wgkernel_cpu` Python 端 microbenchmark，用于建立第一版性能台账。
后续各算子补齐正式 benchmark driver 后，可以用更稳定的结果替换。

| category | op | shape/computing size | dtype | cpu | python/pytorch | cuda | cutlass | note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `activation` | `silu` | `numel=1,048,576` | `fp32` | 1.9531 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
|  | `sigmoid` | `numel=1,048,576` | `fp32` | 1.9619 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
|  | `exp` | `numel=1,048,576` | `fp32` | 1.6279 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
| `elementwise` | `add` | `shape=(1024,1024)` | `fp32` | 0.3674 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
| `embedding / indexing` | `embedding` | `weight=(65536,128), indices=(64,128)` | `fp32` | 0.5747 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
| `gemm / matmul` | `sgemm` | `M=N=K=256` | `fp32` | 8.0030 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
| `norm` | `batchnorm2d_inference_nchw` | `shape=(8,64,56,56)` | `fp32` | 7.2779 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
|  | `rmsnorm` | `shape=(16,512,768)` | `fp32` | 4.9455 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
| `pooling` | `maxpool2d_nchw` | `shape=(8,64,112,112), k=2,s=2` | `fp32` | 4.9341 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |
| `reduce` | `sum` | `numel=16,777,216` | `fp32` | TBD | TBD | 0.2764 ms | TBD | CUDA handwritten v2, [analysis](tests/test_reduce/profiling/reduce_performance_analysis.md) |
|  | `max` | `numel=16,777,216` | `fp32` | TBD | TBD | 0.2766 ms | TBD | CUDA handwritten v2, [analysis](tests/test_reduce/profiling/reduce_performance_analysis.md) |
|  | `argmax` | `numel=16,777,216` | `fp32` | TBD | TBD | 0.2934 ms | TBD | CUDA handwritten v1, [analysis](tests/test_reduce/profiling/reduce_performance_analysis.md) |
|  | `softmax` | `numel=4,194,304` | `fp32` | 16.0550 ms | TBD | TBD | TBD | CPU scalar baseline, correctness passed |

维护规则：

1. 同一个 `op + shape/problem size + dtype` 下，必须先有 `cpu` baseline 行。
2. `cpu`、`cuda`、`cutlass`、`python/pytorch` 列均记录 latency，默认单位为 `ms`。
3. Python 测试脚本会覆盖多个 shape；README 性能表只记录该算子测试集合中最大的
   `shape/computing size`，用于展示最直观的加速比。
4. 完整多规模 benchmark 放到对应算子的 `profiling/` 文档。
5. 加速比由同一行的 `CPU latency / backend latency` 计算；不同 shape 不直接比较。
6. 每次更新性能数字时，同步记录硬件、软件版本和 benchmark 命令。

## 构建方式

项目使用 `CMake` 管理构建。

```bash
cmake -S . -B build
cmake --build build -j
```

常用选项：

- `-DWGKERNEL_BUILD_BENCHMARKS=ON/OFF`
- `-DWGKERNEL_ENABLE_CUDA=ON/OFF`
- `-DCMAKE_CUDA_ARCHITECTURES=native`

测试脚本直接通过 Python 运行，并通过 `--device` 选择后端模块，例如：

```bash
PYTHONPATH=build/python python3 tests/test_reduce/scripts/test_reduce.py --device cuda
PYTHONPATH=build/python python3 tests/test_conv2d/scripts/test_conv2d.py --device cuda
```

## 当前实现进度

> 状态说明：
> `Not Started`：尚未开始
> `In Progress`：正在实现
> `Done`：已完成并通过基本验证

| 类别 | 代表算子 | 当前状态 | 已实现 | 待实现 |
| --- | --- | --- | --- | --- |
| `gemm / matmul` | `sgemm`、`hgemm`、`batched gemm` | `Not Started` | - | `sgemm`、`hgemm`、`batched gemm` |
| `gemv` | `sgemv`、`batched gemv` | `Not Started` | - | `sgemv`、`batched gemv` |
| `reduce` | `sum`、`max`、`argmax`、`softmax`、`logsumexp` | `In Progress` | `sum`、`max`、`argmax` | `softmax`、`logsumexp` |
| `norm` | `layernorm`、`rmsnorm`、`groupnorm` | `Not Started` | - | `layernorm`、`rmsnorm`、`groupnorm` |
| `embedding / indexing` | `embedding`、`gather`、`scatter` | `Not Started` | - | `embedding`、`gather`、`scatter` |
| `activation` | `relu`、`gelu`、`silu`、`sigmoid`、`exp` | `Not Started` | - | `relu`、`gelu`、`silu`、`sigmoid`、`exp` |
| `elementwise` | `add`、`sub`、`mul`、`div`、`bias add` | `Not Started` | - | `add`、`sub`、`mul`、`div`、`bias add` |
| `attention` | `qk matmul`、`masked softmax`、`attention`、`flash attention` | `Not Started` | - | `qk matmul`、`masked softmax`、`attention`、`flash attention` |
| `convolution` | `conv2d`、`depthwise conv`、`im2col + gemm` | `Not Started` | - | `conv2d`、`depthwise conv`、`im2col + gemm` |
| `quantization` | `quantize`、`dequantize`、`int8 gemm`、`fp8 ops` | `Not Started` | - | `quantize`、`dequantize`、`int8 gemm`、`fp8 ops` |
| `layout` | `transpose`、`permute`、`nchw <-> nhwc` | `Not Started` | - | `transpose`、`permute`、`nchw <-> nhwc` |
| `fused ops` | `bias+relu`、`bias+gelu`、`residual+norm` | `Not Started` | - | `bias+relu`、`bias+gelu`、`residual+norm` |

## 后续计划

- 优先完成 `gemm / gemv / reduce / transpose`
- 为每类算子补充最小可运行示例
- 增加统一的测试、benchmark 与 profiling 脚手架
- 逐步补充优化分析文档

## 维护方式

当某个算子完成后，更新上表中的：

- `当前状态`
- `已实现`
- `待实现`

后续也可以继续补充：

- 对应源码路径
- benchmark 结果
- 优化说明
