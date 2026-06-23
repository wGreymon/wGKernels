# Reduce 性能分析

日期：2026-06-22

本文档是 `wGKernel` 中第一份面向算子的性能分析范例，以 CUDA `reduce`
算子为例，建立本仓库统一采用的三层分析方法：

1. 第一层：端到端性能表现，以及相对硬件理论上限的发挥情况。
2. 第二层：基于 Nsight Compute 的 kernel 级 profiling 分析。
3. 第三层：与工业级/参考实现的性能对比。

这份文档的目标不只是记录一次 `reduce` 的测试结果，更重要的是沉淀一套后续可复用
的分析模板，供 `conv2d`、`gemm`、`softmax`、`attention` 等算子沿用。

## 分析范围

本文覆盖当前 CUDA `reduce` 目录下的三个实现：

- `sum`
- `max`
- `argmax`

对应源码：

- `cuda/reduce/src/sum.cu`
- `cuda/reduce/src/max.cu`
- `cuda/reduce/src/argmax.cu`

相关测试与 benchmark 入口：

- `build/benchmarks/cuda/wgkernel_reduce_benchmark`
- `python3 benchmarks/cuda/bench_reduce_torch.py`
- `tests/cuda/test_reduce/scripts/test_reduce_vs_pytorch.py`

## 这份范例希望固定什么

在 `wGKernel` 中，后续每个算子的性能分析都希望按顺序回答下面三个问题。

### 第一层：这个算子整体跑得怎么样

这一层给出用户最直接关心的结果：

- latency
- throughput
- 算子级有效带宽或 FLOP/s
- 相对相关硬件理论上限的利用率

对于 `reduce`，最关键的不是峰值 FP32 算力，而是显存带宽利用率。

### 第二层：为什么会是这个性能

这一层借助 `ncu` 去看主导 kernel 的硬件行为，用 profiling 证据解释第一层的结果，
例如：

- DRAM Throughput
- SM Throughput
- Achieved Occupancy
- Registers/Thread
- 其他与瓶颈相关的指标

### 第三层：和工业实现相比处在什么位置

这一层引入强参考实现，这里用 `PyTorch`，回答：

- 当前实现是否已经有竞争力
- 性能差距主要出现在什么规模
- 差距更像是算法路线问题、架构利用率问题，还是启动开销问题

## 测试口径约定

为了让后续算子之间的性能分析可以横向比较，这份文档约定以下口径：

- 测算子本身，不把文件 IO 和进程启动时间混进结果里。
- 正确性测试和性能测试分开维护。
- 明确记录 shape、dtype、warmup、repeat。
- 明确区分“由 latency 推导出的算子级指标”和“profiler 直接采到的硬件指标”。
- 根据算子类型选择正确的理论上限：
  - memory-bound 算子，对比带宽 roof
  - compute-bound 算子，对比算力 roof

对于 `reduce`，本文使用的算子级推导指标为：

```text
effective_bandwidth = numel * sizeof(float) / latency
```

这个带宽是算子级有效带宽估计，主要统计输入读取流量，不刻意精确建模 workspace
的中间访问以及最终标量写回，因此它更适合用来做算子整体表现判断，而不是替代底层
DRAM 事务统计。

## 测试环境

| 项目 | 数值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute Capability | 8.9 |
| SM 数量 | 24 |
| 最大 SM 时钟 | 3105 MHz |
| 最大显存时钟 | 8001 MHz |
| CUDA toolkit | 13.1 |
| Nsight Compute | 2025.4 |
| PyTorch | 2.11.0+cu130 |
| 运行环境 | WSL2 |

本文使用的理论上限如下：

- 峰值显存带宽：`8001 MHz * 2 * 128 bit / 8 = 256 GB/s`
- 峰值 FP32 算力：
  `24 SM * 128 FP32 lanes/SM * 2 FMA ops/cycle * 3.105 GHz = 19.1 TFLOP/s`

对于 `reduce` 的解释规则是：

- 主要关注显存带宽上限
- 峰值 FP32 算力仅作为补充背景，不作为这个算子的核心评判标准

## 复现实验命令

构建：

```bash
cmake --build build -j
```

wGKernel benchmark：

```bash
build/benchmarks/cuda/wgkernel_reduce_benchmark --op sum    --numel 16777216 --warmup 20 --repeat 200
build/benchmarks/cuda/wgkernel_reduce_benchmark --op max    --numel 16777216 --warmup 20 --repeat 200
build/benchmarks/cuda/wgkernel_reduce_benchmark --op argmax --numel 16777216 --warmup 20 --repeat 200
```

PyTorch benchmark：

```bash
python3 benchmarks/cuda/bench_reduce_torch.py --op sum    --numel 16777216 --warmup 20 --repeat 200
python3 benchmarks/cuda/bench_reduce_torch.py --op max    --numel 16777216 --warmup 20 --repeat 200
python3 benchmarks/cuda/bench_reduce_torch.py --op argmax --numel 16777216 --warmup 20 --repeat 200
```

NCU 采集命令：

```bash
WGKERNEL_NCU_SET=basic \
WGKERNEL_NCU_KERNEL='regex:reduce_.*kernel' \
WGKERNEL_NCU_LAUNCH_SKIP=20 \
WGKERNEL_NCU_LAUNCH_COUNT=1 \
WGKERNEL_NCU_EXPORT="tests/cuda/test_reduce/profiling/reduce_sum_basic" \
./ncu/scripts/profile_reduce.sh build/benchmarks/cuda/wgkernel_reduce_benchmark \
  --op sum --numel 16777216 --warmup 20 --repeat 5
```

`max` 和 `argmax` 只需要替换 `--op` 以及导出路径即可。

## 第一层：端到端性能表现

这一层要回答的问题是：当前 `reduce` 实现整体上把显存系统打到了什么程度。

这里采用的峰值带宽为 `256 GB/s`。

| op | numel | latency_ms | effective_GB/s | % peak bandwidth |
| --- | ---: | ---: | ---: | ---: |
| sum | 1,048,576 | 0.0329 | 127.56 | 49.8% |
| max | 1,048,576 | 0.0202 | 207.71 | 81.1% |
| argmax | 1,048,576 | 0.0306 | 137.22 | 53.6% |
| sum | 16,777,216 | 0.2764 | 242.81 | 94.8% |
| max | 16,777,216 | 0.2766 | 242.64 | 94.8% |
| argmax | 16,777,216 | 0.2934 | 228.70 | 89.3% |
| sum | 67,108,864 | 1.0816 | 248.19 | 96.9% |
| max | 67,108,864 | 1.0812 | 248.28 | 97.0% |
| argmax | 67,108,864 | 1.1303 | 237.49 | 92.8% |

### 第一层解读

- 对于大输入规模，`sum` 和 `max` 已经非常接近带宽上限，大约达到理论峰值的
  95% 到 97%。
- `argmax` 略低一些，但仍然很强，大规模下大约能达到 89% 到 93%。
- 小规模输入更容易受到 kernel 启动开销和多阶段 reduce 开销影响，因此结果波动更大，
  利用率也不够稳定。

### 第一层结论

当前 `reduce` 实现已经是一个很强的 bandwidth-bound baseline。对这类算子，
应该主要看带宽利用率，而不是看它只发挥了多少峰值 FP32 TFLOP/s。

如果只从算力利用率出发，很容易误判这个 kernel 性能不够好，但那其实是拿错了
理论上限。

## 第二层：Nsight Compute Profiling

这一层要回答的问题是：为什么第一层已经接近带宽上限，以及为什么 `argmax`
仍然比 `sum/max` 稍弱一些。

环境说明：当前机器运行在 WSL2 下。早期尝试时曾遇到
`ERR_NVGPUCTRPERM`，在开启 GPU performance counters 权限后，
Nsight Compute 已可以正常采集硬件指标，因此下面引用的 `.ncu-rep`
报告都是真实结果。

归档报告路径：

- `tests/cuda/test_reduce/profiling/reduce_sum_basic.ncu-rep`
- `tests/cuda/test_reduce/profiling/reduce_max_basic.ncu-rep`
- `tests/cuda/test_reduce/profiling/reduce_argmax_basic.ncu-rep`

每份报告都在 `numel = 16,777,216`、`--set basic` 条件下，只抓取一次主导
kernel 启动。

对应的 kernel 为：

- `sum`：`reduce_sum_v2_kernel`
- `max`：`reduce_max_v2_kernel`
- `argmax`：`reduce_argmax_v1_first_stage_kernel`

如需重新查看归档报告，可执行：

```bash
ncu --import tests/cuda/test_reduce/profiling/reduce_sum_basic.ncu-rep --page details
```

关键指标如下：

| op | Duration (us) | DRAM Throughput | Compute (SM) | Achieved Occupancy | Registers/Thread |
| --- | ---: | ---: | ---: | ---: | ---: |
| sum | 271.71 | 96.74% | 8.81% | 98.01% | 32 |
| max | 272.35 | 96.64% | 10.37% | 98.15% | 32 |
| argmax | 289.63 | 90.82% | 20.57% | 96.78% | 38 |

### 第二层解读

- DRAM Throughput 很高，大约在 91% 到 97% 之间，而 SM Throughput 明显更低，
  这是非常典型的 memory-bound 特征。
- Achieved Occupancy 也很高，大约在 97% 到 98% 之间，因此 occupancy
  并不是当前的主要瓶颈。
- `sum` 和 `max` 的 profiling 形态几乎一致，这和第一层中两者带宽利用率接近的
  现象是对应的。
- `argmax` 的画像明显不同：
  - 每线程寄存器更多：`38` 对比 `32`
  - DRAM Throughput 更低：`90.82%`
  - SM Throughput 更高：`20.57%`

这是符合预期的。`argmax` 规约的不是单个标量，而是 `(value, index)` 对，因此
每个元素需要携带更多状态，也需要更多比较和数据搬运，这会自然带来更重的 kernel
主体开销。

### 第二层交叉验证

第一层的带宽估算和第二层的硬件计数器结果是互相印证的：

- 第一层 `sum` 在 `16,777,216` 上的结果：理论峰值带宽的 `94.8%`
- 第二层 `sum` 的 DRAM Throughput：`96.74%`

两者只差几个百分点，这说明 latency 推导出来的算子级判断和 profiler 采到的
硬件级结论是同向的，也说明这套分析口径是可信的。

### 第二层结论

profiling 证据表明，当前 `reduce` kernel 并不是算力受限，而是显存系统受限。
因此后续优化重点应该继续围绕 memory-system efficiency 展开，而 `argmax`
则需要额外关注 pair 状态处理带来的成本。

## 第三层：与 PyTorch 的性能对比

这一层要回答的问题是：当前实现和一个强参考实现相比，已经处在什么水平。

这里采用的参考实现是 `PyTorch`。

定义如下：

```text
speedup_vs_torch = torch_latency / wgkernel_latency
```

数值大于 `1.0x` 表示 `wGKernel` 更快。

| op | numel | wGKernel ms | PyTorch ms | speedup_vs_torch | wGKernel GB/s | PyTorch GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sum | 1,048,576 | 0.0329 | 0.0192 | 0.58x | 127.56 | 218.22 |
| max | 1,048,576 | 0.0202 | 0.0259 | 1.28x | 207.71 | 162.13 |
| argmax | 1,048,576 | 0.0306 | 0.0199 | 0.65x | 137.22 | 210.27 |
| sum | 16,777,216 | 0.2764 | 0.2734 | 0.99x | 242.81 | 245.44 |
| max | 16,777,216 | 0.2766 | 0.2736 | 0.99x | 242.64 | 245.28 |
| argmax | 16,777,216 | 0.2934 | 0.2752 | 0.94x | 228.70 | 243.89 |
| sum | 67,108,864 | 1.0816 | 1.0785 | 1.00x | 248.19 | 248.90 |
| max | 67,108,864 | 1.0812 | 1.2330 | 1.14x | 248.28 | 217.71 |
| argmax | 67,108,864 | 1.1303 | 1.0873 | 0.96x | 237.49 | 246.87 |

### 第三层解读

- 对于大规模输入，`sum` 基本和 PyTorch 打平。
- 对于大规模输入，`max` 已经具备竞争力，在这次测试里甚至快于 PyTorch。
- `argmax` 仍然略慢于 PyTorch，但大规模下的差距已经不算大。
- 更明显的差距主要出现在小规模输入，此时固定开销和启动开销更显著。

### 第三层结论

当前 `reduce` 实现，尤其是 `sum` 和 `max`，已经具备和工业参考实现正面对比的能力。
对于一个仓库中的 baseline 来说，这已经是非常有价值的起点：说明当前实现不仅能跑，
而且在大规模场景下已经接近成熟实现的水平。

## 总结

作为本仓库第一份算子性能分析范例，`reduce` 建立了后续文档希望统一遵循的流程：

1. 先看端到端结果，并用正确的硬件 roof 去解释。
2. 再看 NCU profiling，用 kernel 级证据解释性能来源。
3. 最后和强参考实现对比，判断当前实现的工程位置。

对于 `reduce` 来说，整体结论是：

- `sum` 和 `max` 已经非常接近带宽上限，并且在大规模输入上与 PyTorch 竞争力很强。
- `argmax` 也已经很不错，但仍然要为 pair 状态处理付出额外成本。

## 后续优化方向

1. 给 benchmark 增加版本选择能力，支持直接比较 `reduce` 的不同实现版本。
2. 为 `sum` 和 `max` 增加 `float4` 等 vectorized load 版本，观察是否还能继续逼近
   实际带宽上限。
3. 单独深入分析 `argmax`：
   - pair 布局和对齐方式
   - 是否存在更轻量的内部表示
   - 如何降低小规模输入下的固定开销
4. 后续将这套三层模板迁移到其他算子，但按算子类型切换关注重点：
   - `reduce` / `transpose` / `indexing`：以带宽分析为主
   - `gemm` / `conv`：以算力分析为主
   - `softmax` / `attention`：同时看带宽和算力
