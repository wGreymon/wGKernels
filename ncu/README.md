# Nsight Compute Tutorial

`ncu/` 用于沉淀 NVIDIA Nsight Compute CLI 的使用方法、指标集合、分析脚本、报告摘要与优化结论。

当前本机工具版本：

```bash
ncu --version
```

当前环境输出为 `Nsight Compute Command Line Profiler 2025.4.0.0`。

## 1. NCU 是什么

`ncu` 是 Nsight Compute 的命令行 profiler，用来分析单个 CUDA kernel 的硬件行为。

它和普通 benchmark 的角色不同：

| 工具 | 主要回答的问题 |
| --- | --- |
| CUDA event benchmark | 这个 kernel 跑了多久？吞吐是多少？ |
| `ncu` | 为什么是这个速度？瓶颈在访存、计算、occupancy、还是 warp stall？ |

所以推荐流程是：

1. 先用项目里的 benchmark 得到稳定 latency。
2. 再用 `ncu` 分析目标 kernel。
3. 最后把结论写到 `ncu/notes/` 或算子自己的 `notes.md`。

## 2. 目录约定

```text
ncu/
├── README.md
├── scripts/
├── metric_sets/
├── reports/
└── notes/
```

目录用途：

| 路径 | 用途 |
| --- | --- |
| `ncu/scripts/` | 放可复用 profile 脚本 |
| `ncu/metric_sets/` | 记录不同 kernel 类型关注哪些指标 |
| `ncu/reports/` | 放导出的报告摘要 |
| `ncu/notes/` | 记录分析结论和下一步优化方向 |

原始 `*.ncu-rep` 文件通常比较大，默认不提交到仓库。项目 `.gitignore` 已经忽略 `*.ncu-rep`。

## 3. 基础命令

先确保 benchmark 已经构建：

```bash
cmake -S . -B build
cmake --build build -j
```

先跑普通 benchmark：

```bash
./build/benchmarks/cuda/wgkernel_reduce_benchmark \
    --op sum \
    --numel 16777216 \
    --warmup 10 \
    --repeat 100
```

再用 `ncu` 分析：

```bash
ncu \
    --set basic \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name regex:reduce_sum_v2_kernel \
    --launch-skip 10 \
    --launch-count 1 \
    ./build/benchmarks/cuda/wgkernel_reduce_benchmark \
        --op sum \
        --numel 16777216 \
        --warmup 10 \
        --repeat 100
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `--set basic` | 采集基础 section，速度较快 |
| `--target-processes all` | profile 目标程序和子进程 |
| `--kernel-name-base demangled` | 用可读的 demangled kernel 名称匹配 |
| `--kernel-name regex:...` | 只分析匹配的 kernel |
| `--launch-skip 10` | 跳过前 10 次匹配 launch，避开 warmup |
| `--launch-count 1` | 只采集 1 次匹配 launch，避免报告太大 |

## 4. 导出报告

导出 `.ncu-rep`：

```bash
ncu \
    --set basic \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name regex:reduce_sum_v2_kernel \
    --launch-skip 10 \
    --launch-count 1 \
    --export ncu/reports/reduce_sum_v2_basic \
    --force-overwrite \
    ./build/benchmarks/cuda/wgkernel_reduce_benchmark \
        --op sum \
        --numel 16777216 \
        --warmup 10 \
        --repeat 100
```

读取已有报告：

```bash
ncu --import ncu/reports/reduce_sum_v2_basic.ncu-rep
```

导出 CSV 方便记录：

```bash
ncu \
    --import ncu/reports/reduce_sum_v2_basic.ncu-rep \
    --csv \
    --page details
```

## 5. Section Sets

查看本机支持的 set：

```bash
ncu --list-sets
```

当前常用 set：

| Set | 适合场景 |
| --- | --- |
| `basic` | 初次分析，采集快，足够判断大方向 |
| `detailed` | 需要更多访存、source、roofline 信息 |
| `full` | 最完整，但很慢，只在必要时使用 |
| `roofline` | 判断 compute-bound / memory-bound |

建议顺序：

1. 先用 `basic`。
2. 看不清瓶颈时用 `detailed`。
3. 只在写深入分析报告时用 `full` 或 `roofline`。

## 6. 关键指标怎么看

NCU 的报告会按 section 展示。刚开始不用背所有 metric 名字，先看 section 里的高层结论。

常看的 section：

| Section | 重点 |
| --- | --- |
| `GPU Speed Of Light Throughput` | SM 和 Memory 吞吐接近峰值多少 |
| `Launch Statistics` | grid/block 配置、寄存器、shared memory |
| `Occupancy` | 理论和实际 occupancy |
| `Memory Workload Analysis` | L1/L2/DRAM 访问情况 |
| `Compute Workload Analysis` | 指令吞吐和计算单元利用率 |
| `Scheduler Statistics` | warp 调度情况 |
| `Warp State Statistics` | warp stall 原因 |

经验判断：

| 现象 | 可能瓶颈 |
| --- | --- |
| Memory Throughput 高，SM Throughput 低 | memory-bound |
| SM Throughput 高，Memory Throughput 低 | compute-bound |
| Achieved Occupancy 很低 | block size、寄存器、shared memory 或 launch 配置问题 |
| Warp Stall Long Scoreboard 高 | 等待内存依赖 |
| Warp Stall Barrier 高 | `__syncthreads()` 或同步过多 |
| DRAM Throughput 低但 latency 高 | 访存不合并、访问模式差、cache 命中问题 |

## 7. 分析 Reduce Kernel

以当前 `reduce_sum_v2_kernel` 为例，它是典型 memory-bound kernel。

运行：

```bash
ncu \
    --set basic \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name regex:reduce_sum_v2_kernel \
    --launch-skip 10 \
    --launch-count 1 \
    ./build/benchmarks/cuda/wgkernel_reduce_benchmark \
        --op sum \
        --numel 16777216 \
        --warmup 10 \
        --repeat 100
```

读报告时优先看：

| 关注点 | 为什么 |
| --- | --- |
| Memory Throughput | reduce 主要读全量输入 |
| DRAM Throughput | 判断是否接近显存带宽 |
| L2 Throughput | 判断缓存是否参与较多 |
| Achieved Occupancy | 判断并行度是否足够 |
| Warp Stall Barrier | 对比 `v1` shared-memory reduce 和 `v2` shuffle reduce |
| Warp Stall Long Scoreboard | 判断是否主要卡在访存依赖 |

对比 `v1` 和 `v2` 时，重点不是只看 latency，还要看：

| 对比项 | 预期变化 |
| --- | --- |
| shared memory 访问 | `v2` 应减少 |
| barrier stall | `v2` 应减少 |
| memory throughput | `v2` 可能更高 |
| occupancy | 一般应接近或不变 |

## 8. 常用命令模板

分析 `sum`：

```bash
ncu --set basic --target-processes all --kernel-name-base demangled \
    --kernel-name regex:reduce_sum_v2_kernel --launch-skip 10 --launch-count 1 \
    ./build/benchmarks/cuda/wgkernel_reduce_benchmark --op sum --numel 16777216
```

分析 `max`：

```bash
ncu --set basic --target-processes all --kernel-name-base demangled \
    --kernel-name regex:reduce_max_v2_kernel --launch-skip 10 --launch-count 1 \
    ./build/benchmarks/cuda/wgkernel_reduce_benchmark --op max --numel 16777216
```

分析 `argmax`：

```bash
ncu --set basic --target-processes all --kernel-name-base demangled \
    --kernel-name regex:reduce_argmax_v1_first_stage_kernel --launch-skip 10 --launch-count 1 \
    ./build/benchmarks/cuda/wgkernel_reduce_benchmark --op argmax --numel 16777216
```

使用项目脚本：

```bash
ncu/scripts/profile_reduce.sh \
    ./build/benchmarks/cuda/wgkernel_reduce_benchmark \
    --op sum \
    --numel 16777216 \
    --warmup 10 \
    --repeat 100
```

脚本默认使用：

| 环境变量 | 默认值 |
| --- | --- |
| `WGKERNEL_NCU_SET` | `basic` |
| `WGKERNEL_NCU_KERNEL` | `regex:reduce_.*kernel` |
| `WGKERNEL_NCU_LAUNCH_SKIP` | `10` |
| `WGKERNEL_NCU_LAUNCH_COUNT` | `1` |

指定导出路径：

```bash
WGKERNEL_NCU_EXPORT=ncu/reports/reduce_sum_v2_basic \
WGKERNEL_NCU_KERNEL=regex:reduce_sum_v2_kernel \
ncu/scripts/profile_reduce.sh \
    ./build/benchmarks/cuda/wgkernel_reduce_benchmark \
    --op sum \
    --numel 16777216 \
    --warmup 10 \
    --repeat 100
```

## 9. 写分析结论

建议每次 profile 后记录这些内容：

```text
Kernel:
Input shape:
GPU:
CUDA:
NCU:
Benchmark latency:
NCU set:

Main observation:
Bottleneck:
Evidence:
Next action:
```

示例：

```text
Kernel: reduce_sum_v2_kernel
Input shape: 16777216 float32 elements
Benchmark latency: 0.2765 ms
NCU set: basic

Main observation:
The kernel behaves like a memory-bound reduction.

Bottleneck:
Global memory throughput and memory dependency stalls dominate.

Evidence:
Memory throughput is much higher than SM compute throughput.
The kernel performs very little arithmetic per loaded element.

Next action:
Try vectorized loads and unrolling to improve memory instruction efficiency.
```

## 10. 常见问题

`ncu` 比 benchmark 慢很多：

这是正常的。NCU 会重放 kernel 或采集额外硬件计数器，不应该用 NCU 的运行时间替代 benchmark latency。

报告里 kernel 太多：

使用 `--kernel-name`、`--launch-skip`、`--launch-count` 过滤。

提示没有权限访问 performance counter：

这通常是系统限制了 GPU performance counter。需要在宿主机或驱动配置里开放相关权限。

找不到源码行：

需要构建时保留调试信息或 line info。后续可以给 CUDA 编译选项加 `-lineinfo`，用于 source-level profiling。

多个 benchmark 同时跑导致结果异常：

profile 和 benchmark 都应该单独运行。并行执行多个 GPU benchmark 会互相抢资源，结果不能作为性能结论。

## 11. 官方资料

- [NVIDIA Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
