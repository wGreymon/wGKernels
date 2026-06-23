# NVIDIA Nsight Compute 完整教程

> Nsight Compute (NCU) 是 NVIDIA 的 GPU kernel 级别性能分析工具，用于深入理解 kernel 的硬件行为、定位瓶颈并指导优化方向。

## 目录

1. [基础概念](#1-基础概念)
2. [安装与环境](#2-安装与环境)
3. [核心命令](#3-核心命令)
4. [ncu-ui 可视化界面](#4-ncu-ui-可视化界面)
5. [Metric Sets 与 Sections](#5-metric-sets-与-sections)
6. [关键指标解读](#6-关键指标解读)
7. [瓶颈诊断流程](#7-瓶颈诊断流程)
8. [常用场景示例](#8-常用场景示例)
9. [高级用法](#9-高级用法)
10. [常见问题](#10-常见问题)

---

## 1. 基础概念

### 1.1 NCU vs 其他工具

| 工具 | 回答的问题 |
|------|-----------|
| `nvcc` 编译 + 运行 | kernel 跑出来了吗 |
| CUDA Events / chrono benchmark | kernel 跑了多久 |
| `nvprof` / Nsight Systems | 整个 CUDA 程序的 timeline |
| **`ncu` Nsight Compute** | **为什么是这个速度，瓶颈在哪** |

NCU 的核心价值：**把 latency 拆解成硬件原因**。

### 1.2 工作原理

NCU 通过 GPU 硬件性能计数器（hardware performance counters）采集数据。有两种采集模式：

- **Single-pass**：部分指标可以在一次 kernel 执行中同时采集
- **Multi-pass**：某些指标互相冲突，需要多次执行 kernel 分别采集

NCU 报告中的 "Estimated Metrics" 数量反映了这个开销——数量越多意味着越慢。

### 1.3 核心概念

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU Speed Of Light                       │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ SM Compute│   │  Memory      │   │  L2 Cache          │  │
│  │ Throughput│   │  Throughput   │   │  Throughput         │  │
│  └──────────┘   └──────────────┘   └────────────────────┘  │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ Achieved │   │  Warp Stall   │   │  Source Counters   │  │
│  │ Occupancy│   │  Breakdown     │   │  (line-level)      │  │
│  └──────────┘   └──────────────┘   └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

- **Throughput**：设备在某个维度上的最大利用程度（接近 100% = 逼近硬件上限）
- **Occupancy**：SM 上活跃 warp 的比例
- **Stall**：warp 因某种原因暂停执行，等待资源

### 1.4 分析流程

```
1. 用 benchmark 测出 latency
   ↓
2. 用 ncu --set basic 快速定位大方向
   ↓
3. 根据方向选择 detailed / roofline 深入
   ↓
4. 根据瓶颈类型选择优化方向
   ↓
5. 改代码 + benchmark 验证 + ncu 确认
```

---

## 2. 安装与环境

### 2.1 检查是否安装

```bash
ncu --version
```

Nsight Compute 通常随 NVIDIA CUDA Toolkit 一起安装，或独立从 [NVIDIA 官网](https://developer.nvidia.com/nsight-compute) 下载。

### 2.2 WSL2 环境权限问题

在 WSL2 中运行 NCU 可能遇到：

```
==ERROR== ERR_NVGPUCTRPERM: The user does not have permission to profile this context.
```

**解决方案**：在 **Windows 宿主机**上以管理员权限运行：

```powershell
# 管理员 PowerShell
nvidia-smi -c 0           # 允许所有用户访问 GPU计数器
# 或
nvidia-smi acci=0         # 精确控制
```

然后在 WSL2 中重新运行 NCU。

### 2.3 查看可用 GPU

```bash
nvidia-smi
# 或
ncu --query-gpu
```

---

## 3. 核心命令

### 3.1 最简形式

```bash
ncu [options] ./your_cuda_program [program_args]
```

### 3.2 常用参数

```bash
# 指定 metric set
ncu --set basic ./program

# 只 profile 特定 kernel（支持正则）
ncu --kernel-name regex:reduce_.*kernel ./program

# 跳过前 N 次 launch（避开 warmup）
ncu --launch-skip 10 ./program

# 只采集 N 次 launch（避免报告过大）
ncu --launch-count 1 ./program

# 导出报告
ncu --export ./report_name --force-overwrite ./program

# 使用 demangled（可读）的 kernel 名称
ncu --kernel-name-base demangled ./program

# 指定 GPU
ncu --gpu <index> ./program

# 只显示特定 section
ncu --section SpeedOfLight ./program

# 生成 CSV 格式输出
ncu --csv ./program
```

### 3.3 读取已有报告

```bash
# 交互式查看报告
ncu --import ./report.ncu-rep

# 带过滤的查看
ncu --import ./report.ncu-rep --page details
ncu --import ./report.ncu-rep --section SpeedOfLight
ncu --import ./report.ncu-rep --csv > report.csv
```

### 3.4 过滤多个 kernel

```bash
# profile 所有包含 "gemm" 或 "reduce" 的 kernel
ncu --kernel-name regex:gemm|reduce ./program

# profile 所有带特定前缀的 kernel
ncu --kernel-name regex:wgkernel::.* ./program
```

### 3.5 指定 section 组合

```bash
# 自定义 section 组合
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    ./program
```

查看所有可用 section：

```bash
ncu --list-sections
```

---

## 4. ncu-ui 可视化界面

Nsight Compute 包含两个组件：

| 组件 | 用途 |
|------|------|
| `ncu` | 命令行 profiler，生成 `.ncu-rep` 报告文件 |
| `ncu-ui` | GUI 可视化工具，读取 `.ncu-rep` 进行交互式分析 |

两者配合使用：**CLI 生成报告，GUI 查看分析**。

### 4.1 启动 GUI

```bash
# 直接启动（自动打开 .ncu-rep 文件）
ncu-ui ./report.ncu-rep

# 或先启动 GUI，再通过菜单打开
ncu-ui
```

如果 `ncu-ui` 找不到，先检查是否安装：

```bash
which ncu-ui
# 通常路径：/usr/local/cuda/bin/ncu-ui
# 或：/opt/nvidia/nsight-compute/*/ncu-ui
```

### 4.2 GUI 界面布局

```
┌──────────────────────────────────────────────────────────────────┐
│  File  Edit  View  Analysis  Window  Help                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐  ┌───────────────────────────────────┐ │
│  │  Kernels            │  │  Speed Of Light                   │ │
│  │  ├─ reduce_sum_v2   │  │  ┌─────────────────────────────┐ │ │
│  │  ├─ reduce_max_v1   │  │  │  SM Throughput   ████░░ 61%  │ │ │
│  │  └─ reduce_argmax   │  │  │  Memory Throughput ██████ 97% │ │ │
│  │                     │  │  └─────────────────────────────┘ │ │
│  │  ┌─────────────────┐│  │                                    │
│  │  │  Sections        ││  │  Occupancy                         │
│  │  │  ├─ Launch Stats ││  │  ┌─────────────────────────────┐ │ │
│  │  │  ├─ SpeedOfLight ││  │  │  ████████████████░░  84%     │ │ │
│  │  │  ├─ Occupancy    ││  │  └─────────────────────────────┘ │ │
│  │  │  ├─ Memory       ││  │                                    │
│  │  │  └─ Warp States   ││  │  Warp Stall Breakdown              │
│  │  └─────────────────┘│  │  ┌─────────────────────────────┐ │ │
│  │                     │  │  │  Long Scoreboard  ████████  │ │ │
│  └─────────────────────┘  │  │  Barrier           █░░░░░░  │ │ │
│                           │  └─────────────────────────────┘ │ │
│                           └───────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│  Status: 1 kernel, 1 pass, GPU: RTX 4060 Laptop GPU (sm_89)     │
└──────────────────────────────────────────────────────────────────┘
```

左侧导航栏：
- **Kernels** — 如果报告包含多个 kernel，在此切换
- **Sections** — 按 section 分组查看指标

右侧主面板：
- 可折叠的 section 卡片，带进度条和百分比
- 点击任意指标可查看详细说明

### 4.3 核心视图

#### Speed Of Light（最重要）

用条形图直观展示 SM 和 Memory 吞吐接近峰值的程度。

- Memory Throughput 接近 100% → **memory-bound**
- SM Throughput 接近 100% → **compute-bound**

#### Occupancy

显示理论 vs 实际 occupancy，以及瓶颈因素（Registers / Shared Memory / Warps）。

#### Warp State Statistics

堆叠条形图展示 warp stall 原因分布：
- Long Scoreboard（内存等待）
- Barrier（同步等待）
- 其他

鼠标悬停查看具体百分比。

#### Roofline Chart

只在 `--set roofline` 采集时有此 section。

交互式 roofline 图，可放大缩小，标注当前 kernel 落在哪个区域。

### 4.4 对比多个报告

GUI 支持同时打开多个 `.ncu-rep` 文件进行对比：

```bash
# 方式一：命令行同时打开
ncu-ui ./report_v1.ncu-rep ./report_v2.ncu-rep

# 方式二：GUI 菜单 → File → Open Additional Report
```

对比视图会并排显示同一个指标在两个报告中的差异：

```
                          report_v1    report_v2    Delta
SM Throughput              8.81%        12.34%     +3.53%
Memory Throughput          96.74%       94.21%     -2.53%
Occupancy                  98.01%       98.15%     +0.14%
Warp Stall Long Scoreboard 88.93%       82.10%     -6.83%
```

### 4.5 Source 视图（需要 -lineinfo）

如果 kernel 编译时带了 `-lineinfo`，GUI 可以显示源码级分析：

1. 在左侧选择 **Source** tab
2. 左侧列出 kernel 的源码文件
3. 右侧显示各行代码的耗时 heatmap

颜色含义：
- 红色 = 热点行，耗时最多
- 蓝色 = 较少耗时
- 灰色 = 该行无采样数据

### 4.6 常用 GUI 操作

| 操作 | 方法 |
|------|------|
| 打开报告 | `ncu-ui ./report.ncu-rep` 或 `File → Open` |
| 切换 kernel | 左侧 Kernels 栏点击 |
| 切换 section | 左侧 Sections 栏点击 |
| 导出当前视图为图片 | `File → Export Current View` |
| 复制指标数值 | 右键 → Copy Value |
| 查看指标说明 | 鼠标悬停指标名称 |
| 搜索指标 | `Ctrl+F` 搜索指标名 |
| 对比两个报告 | `File → Open Additional Report` |
| 重置视图 | `View → Reset Layout` |
| 全屏 | `F11` |

### 4.7 CLI vs GUI 怎么选

| 场景 | 推荐工具 |
|------|---------|
| 快速循环开发（改代码 → profile → 改代码） | CLI + `--import --section` |
| 第一次分析新 kernel | **GUI**，直观探索 |
| 需要对比两个版本的差异 | **GUI**，并排对比 |
| 生成正式报告文档 | CLI `--csv` + 脚本后处理 |
| CI / 自动化 pipeline | CLI，绝对不用 GUI |
| 查看 source-level heatmap | **GUI**，交互体验好 |

**推荐工作流**：CLI 生成报告 → GUI 深度分析 → CLI 批量提取数据写文档。

### 4.8 报告文件格式

`.ncu-rep` 实际是一个**目录**（类似 `.tar` 的打包结构）：

```bash
# 查看内部结构
ls -la ./report.ncu-rep/

# 解压查看
unzip -l ./report.ncu-rep
```

包含的文件：
- `metadata.json` — 采集环境信息（GPU、CUDA 版本、timestamp）
- `metric_database.pb` — 实际指标数据（protobuf 格式）
- `source_info/` — 源码映射（如果有 line info）

可以用 Python 直接解析（需要 nvidia pickle 库或第三方工具），但更简单的方式还是用 `ncu --import`。

---

## 5. Metric Sets 与 Sections

### 4.1 Metric Sets（采集集）

```bash
ncu --list-sets
```

| Set | Metrics 数量 | 速度 | 适用场景 |
|-----|------------|------|---------|
| `basic` | ~213 | 快 | **初次分析，通用场景** |
| `detailed` | ~996 | 中 | 需要 source、roofline、更多访存信息 |
| `full` | ~8054 | 慢 | 深度分析，最完整的指标 |
| `roofline` | ~6679 | 慢 | 生成 roofline chart，判断 compute/memory bound |
| `pmsampling` | ~553 | 中 | warp stall 采样分析 |
| `nvlink` | ~122 | 快 | 多 GPU NVLink 拓扑分析 |

**推荐流程**：

```
basic → (如果需要更多细节) → detailed → (深度优化) → full/roofline
```

### 4.2 Sections（报告分区）

每个 set 包含若干 section。独立指定时用 `--section`：

```bash
# 最常用的 section 组合
ncu --section SpeedOfLight \
    --section Occupancy \
    --section LaunchStats \
    --section MemoryWorkloadAnalysis \
    --section WarpStateStats \
    ./program
```

#### Launch Statistics

kernel 启动配置信息——**每次必看**。

```text
Kernel Latency                                1.272 us
Grid Size                                     65536
Block Size                                    256
Registers Per Thread                          32
Shared Memory Configuration Size (bytes)       0
Driver Shared Memory Per Block (bytes)         0
Device Memory                              4096 bytes
```

关注点：
- Grid Size / Block Size 是否合理
- 寄存器数量（影响 occupancy）
- Shared Memory 使用量

#### GPU Speed Of Light

**最核心的 section**，直接告诉你设备在各个维度上被用了多少。

```text
Achieved SM Frequency                         3105 MHz
Teoretical SM Frequency                      2475 MHz
SM Active Warp Time Per Edge Tick (%)        91.22%

Compute (SM) Throughput                      8.81 %
Memory Throughput                            96.74 %
L2 Throughput                                96.78 %
```

**经验法则**：

| 现象 | 结论 |
|------|------|
| Memory Throughput 高，Compute Throughput 低 | **memory-bound** |
| Compute Throughput 高，Memory Throughput 低 | **compute-bound** |
| 两者都低 | 可能是 launch 配置或同步问题 |
| L2 Throughput >> DRAM Throughput | 数据重用好，命中 L2 |

#### Occupancy

```text
Theoretical Occupancy                         100%
Active Threads Per Multiprocessor             2048
Occupancy                                     98.01%
Block Limit Registers                         1
Block Limit Shared Memory                     1
Block Limit Warps                             1
```

Occupancy 影响因素（按常见程度）：
1. **Registers Per Thread**（最常见）——编译期决定
2. **Shared Memory Per Block**
3. **Block Size**（launch 配置）

Occupancy 高 ≠ 性能好，但 Occupancy 低通常是问题信号。

#### Memory Workload Analysis

显存访问详细分解。

```text
Global Load Efficiency                        100.00%
Global Store Efficiency                        100.00%
L2 Compression Efficiency                      85.23%
L2 Utilization                                 96.78%
DRAM Utilization                               96.74%
```

关键指标：
- **Global Load/Store Efficiency**：实际传输的数据量 vs 理论上最少的请求量。低于 100% 说明访问不合并或对齐不理想。
- **L2 Hit Rate**：高命中率意味着好的数据局部性
- **DRAM Utilization**：接近峰值说明是带宽瓶颈

#### Warp State Statistics

**warp 暂停原因分析**，定位 stall 根源。

```text
Warp Cycles Per Active Warp Cycle (%)         91.22%
Average Active Warps Per Multiprocessor       64.00

Warp Cycles By State:
  Not Selected                                 8.78%
  Selected But Waiting                        91.22%
    - Long Scoreboard                         88.93%  ← 等待内存
    - Short Scoreboard                          0.00%
    - Barrier                                  0.00%
    - Wait For Register Available              0.00%
    - Instruction Fetch                         0.00%
    - Immediate Postiche                       0.00%
    - Device Memory                            0.00%
    - Texture                                   0.00%
    - Synchronization                           0.00%
    - Dispatch Stall                            0.00%
    - Selected but Disabled                    0.00%
```

**Stall 类型解读**：

| Stall 类型 | 含义 | 优化方向 |
|-----------|------|---------|
| Long Scoreboard | 等待 L1/L2/DRAM 内存读完成 | 合并访问、预取、数据布局 |
| Short Scoreboard | 等待寄存器操作完成 | 少见，通常不用担心 |
| Barrier | `__syncthreads()` 等待 | 减少 barrier、重排同步 |
| Wait For Register | 等待寄存器可用 | 减少寄存器压力 |
| Instruction Fetch | 等待取指 | 罕见，可能是分支过多 |
| Dispatch Stall | SM 调度器问题 | 非常罕见 |

#### Scheduler Statistics

```text
No Instructions Executed                      2.01%
Instructions Executed                         97.99%
IPC                                          0.58
```

IPC (Instructions Per Cycle) 低于预期？结合 Warp State Stats 一起看。

---

## 6. 关键指标解读

### 5.1 Memory-bound vs Compute-bound

这是第一个要判断的问题。

**方法一**：直接看 Speed Of Light

```
Memory Throughput:  96.74%  ← 高
Compute Throughput:  8.81%  ← 低
结论: memory-bound
```

**方法二**：用 roofline set

```bash
ncu --set roofline --export roofline_report --force-overwrite ./program
ncu --import roofline_report.ncu-rep
```

Roofline chart 会画出一条带屋顶的图，kernel 的实际算术强度落在哪个区域，一目了然。

### 5.2 瓶颈定位决策树

```
START
  │
  ├─ SpeedOfLight.SM Throughput 高？
  │     ├─ YES → Compute-bound → 往 compute 方向优化
  │     │         - 减少浪费的算术运算
  │     │         - 用更低精度的 math intrinsic
  │     │         - Tensor Core (FP16/BF16)
  │     │
  │     └─ NO ↓
  │
  ├─ SpeedOfLight.Memory Throughput 高？
  │     ├─ YES → Memory-bound → 往带宽方向优化
  │     │         - 提高 DRAM 访问效率
  │     │         - 向量化加载 (float4)
  │     │         - 改善合并访问
  │     │         - 增加数据重用 (tiling)
  │     │
  │     └─ NO ↓
  │
  └─ Occupancy 低？
        ├─ YES → Launch 配置问题
        │         - 调整 block size
        │         - 减少寄存器使用 (-maxrregcount)
        │         - 减少 shared memory 使用
        │
        └─ NO → Warp Stall 分析
                  ├─ Long Scoreboard → 内存依赖
                  ├─ Barrier → 同步过多
                  └─ 其他 → 具体分析
```

### 5.3 常用指标速查表

| 指标 | 好的值 | 差的值 | 优化方向 |
|------|--------|--------|---------|
| Memory Throughput | > 90% | < 60% | 合并访问、向量加载 |
| DRAM Utilization | > 90% | < 60% | 同上 |
| L2 Hit Rate | > 50% | < 20% | 数据重用、tiling |
| Global Load Efficiency | 100% | < 100% | 对齐、合并 |
| Compute Throughput | > 70% | < 30% | 算法优化、低精度 |
| Achieved Occupancy | > 80% | < 50% | 寄存器/shmem/launch |
| Warp Stall Long Scoreboard | - | > 80% | 内存优化 |
| Warp Stall Barrier | - | > 10% | 减少同步 |

---

## 7. 瓶颈诊断流程

### 6.1 快速诊断（5 分钟）

```bash
# 1. 跑 basic set
ncu --set basic \
    --kernel-name regex:YOUR_KERNEL \
    --launch-skip 10 \
    --launch-count 1 \
    --export ./quick_report \
    --force-overwrite \
    ./benchmark --args

# 2. 看这三个数字
ncu --import ./quick_report.ncu-rep --section SpeedOfLight
```

如果 Memory Throughput > 90% 且 Compute Throughput < 30%，**基本确定是 memory-bound**。

### 6.2 深度诊断（30 分钟）

```bash
# 1. detailed set（包含 roofline、memory 更多细节）
ncu --set detailed \
    --kernel-name regex:YOUR_KERNEL \
    --launch-skip 10 \
    --launch-count 1 \
    --export ./deep_report \
    --force-overwrite \
    ./benchmark --args

# 2. 分析关键 sections
ncu --import ./deep_report.ncu-rep --section MemoryWorkloadAnalysis
ncu --import ./deep_report.ncu-rep --section WarpStateStats
```

### 6.3 Source-level 分析

需要编译时加 `-lineinfo`：

```bash
nvcc -lineinfo -Xcompiler -g -arch=sm_89 your_kernel.cu -o benchmark
```

然后用 detailed/full set 分析 Source Counters section，可以看到具体哪一行代码占用最多时间。

```bash
ncu --set detailed \
    --kernel-name regex:YOUR_KERNEL \
    --section SourceCounters \
    --launch-skip 10 \
    --launch-count 1 \
    ./benchmark --args
```

---

## 8. 常用场景示例

### 7.1 分析 Reduce Kernel

Reduce 是典型的 memory-bound kernel：

```bash
ncu --set basic \
    --kernel-name-base demangled \
    --kernel-name regex:reduce_sum_v2_kernel \
    --launch-skip 20 \
    --launch-count 1 \
    ./wgkernel_reduce_benchmark --op sum --numel 16777216 --warmup 20 --repeat 100
```

预期结果：
```
Memory Throughput: ~95%  ✓ memory-bound 特征
Compute Throughput: ~9%  ✓ 算术少
Occupancy: ~98%           ✓ 并行度足够
Warp Stall Long Scoreboard: ~90%  ✓ 等待内存
```

优化方向：向量加载 (`float4`)、unroll、手动展开。

### 7.2 分析 GEMM

GEMM 是 compute-bound 为主的 kernel（接近 roofline）：

```bash
ncu --set roofline \
    --kernel-name regex:gemm_kernel \
    --launch-skip 10 \
    --launch-count 1 \
    --export ./gemm_roofline \
    --force-overwrite \
    ./benchmark --M 4096 --N 4096 --K 4096
```

预期结果：
```
Compute Throughput: 高（接近 FP32 roof）
Memory Throughput: 中等（算术强度高）
```

GEMM 的优化方向：Tensor Core、shared memory tiling、bank conflict 优化。

### 7.3 分析 Conv2d

Conv 混合了 memory-bound 和 compute-bound 特征：

```bash
ncu --set detailed \
    --kernel-name regex:conv2d_.*kernel \
    --launch-skip 10 \
    --launch-count 1 \
    --export ./conv2d_detailed \
    --force-overwrite \
    ./benchmark --algo im2col_gemm --B 1 --C 64 --H 224 --W 224 --K 64 --R 3 --S 3
```

### 7.4 对比两个 kernel 版本

```bash
# profile v1
ncu --set basic --kernel-name regex:v1_kernel --launch-skip 10 --launch-count 1 \
    --export ./v1_report --force-overwrite ./benchmark --version v1

# profile v2
ncu --set basic --kernel-name regex:v2_kernel --launch-skip 10 --launch-count 1 \
    --export ./v2_report --force-overwrite ./benchmark --version v2

# 对比关键指标
echo "=== v1 ===" && ncu --import ./v1_report.ncu-rep --section SpeedOfLight
echo "=== v2 ===" && ncu --import ./v2_report.ncu-rep --section SpeedOfLight
```

### 7.5 分析多个 shape

```bash
for numel in 1048576 16777216 67108864; do
    ncu --set basic \
        --kernel-name regex:reduce_.*kernel \
        --launch-skip 10 --launch-count 1 \
        --export ./reduce_${numel} --force-overwrite \
        ./benchmark --numel $numel --warmup 10 --repeat 50
done
```

---

## 9. 高级用法

### 8.1 自定义指标

通过 `--metrics` 指定要采集的具体指标：

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained,\
               dram__bytes.sum,\
               sm__warps_active.avg.pct_of峰值 \
    ./benchmark
```

查看所有可用指标：

```bash
ncu --query-metrics
```

### 8.2 自定义 Section 组合

创建配置文件 `my_sections.toml`：

```toml
[sections]
# 只包含你最关心的 sections
SpeedOfLight = true
Occupancy = true
MemoryWorkloadAnalysis = true
WarpStateStats = true
```

```bash
ncu --config my_sections.toml ./benchmark
```

### 8.3 后处理脚本

结合 `ncu --import --csv` 生成可读表格：

```bash
# 提取关键指标到 CSV
ncu --import ./report.ncu-rep \
    --csv \
    --section SpeedOfLight \
    | grep -E "Throughput|Occupancy" \
    > metrics.csv
```

### 8.4 Batch Profile

用 `ncu-ui`（GUI 版本）可以同时查看多个报告、对比差异。

### 8.5 限制采集时间

```bash
# 最多 profile 60 秒
ncu --timeout 60 ./benchmark
```

### 8.6 只采集启动统计（最快）

```bash
ncu --section LaunchStats ./benchmark
```

这几乎不会增加 overhead，适合快速检查 launch 配置。

---

## 10. 常见问题

### Q1: ncu 比 benchmark 慢很多，正常吗？

**正常**。NCU 采集硬件计数器需要额外开销，`--set full` 可能让 kernel 慢 10-100 倍。

> **不要用 NCU 的运行时间作为性能数据**。用它来分析瓶颈，用 benchmark 的实际 latency 来衡量性能。

### Q2: 提示 "ERR_NVGPUCTRPERM" 没有权限

在 Windows 宿主机上以管理员权限运行：
```powershell
nvidia-smi -c 0
```
然后在 WSL2 中重新运行 ncu。

### Q3: 报告里没有源码行信息

确保编译时添加了 `-lineinfo`：

```bash
nvcc -lineinfo -Xcompiler -g -arch=sm_89 your_kernel.cu -o benchmark
```

### Q4: kernel 太多，报告太长

用 `--kernel-name`、`--launch-skip`、`--launch-count` 严格过滤：

```bash
ncu --kernel-name regex:your_kernel \
    --launch-skip 10 \
    --launch-count 1 \
    ./benchmark
```

### Q5: occupancy 100% 但还是慢？

Occupancy 100% 只说明 SM 上 warp 足够多，不代表这些 warp 都能往前跑。

可能的原因：
- 所有 warp 都在等待同一个资源（memory dependency）
- 分支分化（branch divergence）导致有效并行度低
- barrier 阻塞

结合 Warp State Stats 诊断。

### Q6: multi-pass 指标采集太慢

某些指标需要多次 kernel 执行。可以用 `--passes` 控制：

```bash
ncu --set detailed --passes 2 ./benchmark
```

### Q7: 如何判断我的优化是否有效？

```
优化前 → benchmark latency
         ↓
       ncu profile → 记录关键指标（Memory Throughput, Occupancy, Stall%）
         ↓
       应用优化
         ↓
       benchmark latency → 验证是否有提升
         ↓
       ncu profile → 确认指标变化是否符合预期
```

---

## 附录：常用命令速查

```bash
# 查看版本
ncu --version

# 查看可用 sets
ncu --list-sets

# 查看可用 sections
ncu --list-sections

# 查看可用 metrics
ncu --query-metrics

# 快速 profile（basic set）
ncu --set basic ./benchmark

# 指定 kernel
ncu --kernel-name regex:kernel_name ./benchmark

# 跳过 warmup
ncu --launch-skip 10 ./benchmark

# 导出报告
ncu --export ./report --force-overwrite ./benchmark

# 读取报告
ncu --import ./report.ncu-rep

# CSV 导出
ncu --import ./report.ncu-rep --csv

# 只看某个 section
ncu --import ./report.ncu-rep --section SpeedOfLight
```

---

## 推荐阅读

- [NVIDIA Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [CUDA Best Practices Guide - Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
