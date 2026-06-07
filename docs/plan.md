# Project Plan

`wGKernel` 的长期目标是沉淀一个面向通用高性能算子工程师的作品集：既覆盖常见算子，也能展示 CUDA、CPU、Tensor Core、CUTLASS、profiling 和 benchmark 的完整工程能力。

## Direction

当前项目分为三条主线推进：

| 主线 | 目标 | 重点 |
| --- | --- | --- |
| 算子覆盖 | 补齐大模型推理和通用深度学习常见算子 | `reduce`、`activation`、`transpose`、`norm`、`softmax`、`gemm`、`attention` |
| 性能分析 | 建立可解释、可复现的 benchmark / profiling 流程 | PyTorch 对齐、cuBLAS/cuDNN/FlashAttention 对齐、Nsight Compute、roofline |
| 深度优化 | 进入矩阵类算子的工业级优化路径 | Tensor Core、WMMA、MMA PTX、CUTLASS、fusion |

## Near-Term Plan

1. 完成 `reduce` 类算子的基础闭环。
2. 补齐 `activation / elementwise`，优先实现 `relu`、`silu`、`gelu`。
3. 补齐 `transpose / layout transform`，重点学习 shared memory tile 和 bank conflict。
4. 实现 `norm` 类算子，优先 `layernorm` 和 `rmsnorm`。
5. 实现 `softmax`，作为 attention 前置能力。
6. 建立更统一的 benchmark 输出格式，包括 latency、GB/s、GOPS、GFLOP/s、speedup。

## Tensor Core Plan

Tensor Core 不作为独立算子目录，而是作为 `gemm`、`convolution`、`attention` 等算子的实现技术路线。

计划路径：

1. 在 `cuda/gemm/src/sgemm.cu` 中保留 CUDA Core 路线的 FP32 GEMM。
2. 新增 `cuda/gemm/src/hgemm_wmma.cu`，使用 WMMA API 实现第一个 Tensor Core HGEMM。
3. 对比 CUDA Core GEMM 和 WMMA HGEMM 的 latency、TFLOPS 和 Nsight Compute 指标。
4. 学习 fragment、warp tile、matrix layout、accumulator、mixed precision accumulation。
5. 新增 `cuda/gemm/src/hgemm_mma.cu`，使用 MMA PTX 或 inline PTX 实现更底层的 Tensor Core GEMM。
6. 学习 `ldmatrix`、`mma.sync`、shared memory swizzle、double buffering / ping-pong buffer。
7. 将 Tensor Core 编程经验迁移到 attention 的 QK / PV matmul 和 convolution 的 implicit GEMM。

重点记录：

- Tensor Core 是否真的被使用。
- Tensor Core utilization。
- shared memory bank conflict。
- global memory load/store efficiency。
- occupancy 与 register pressure。
- 不同 dtype 的性能差异，例如 FP16、BF16、TF32、INT8。

## CUTLASS Plan

CUTLASS 作为工业级 GEMM / Conv / Tensor Core 实现的重要参考，不急于一开始引入完整源码。

计划路径：

1. 先阅读 CUTLASS GEMM examples，理解它如何组织 threadblock tile、warp tile、instruction tile。
2. 在 `docs/cutlass.md` 中记录 CUTLASS 的核心抽象和源码阅读笔记。
3. 当手写 WMMA / MMA 版本完成后，再新增 `cuda/gemm/src/hgemm_cutlass.cu`。
4. 如需依赖 CUTLASS 编译示例，再引入 `third_party/cutlass/`。
5. 实现 CUTLASS GEMM baseline，对齐 cuBLAS / cuBLASLt。
6. 学习 CUTLASS epilogue fusion，实现 `gemm + bias + activation`。
7. 后续学习 CUTLASS 3.x / CuTe，用于理解更现代的 layout 和 pipeline 抽象。

重点记录：

- CUTLASS kernel 配置参数。
- tile shape 对性能的影响。
- epilogue fusion 的收益。
- 与手写 Tensor Core kernel 的差异。
- 与 cuBLAS / cuBLASLt 的性能差距。

## Milestones

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| Stage 1 | CUDA reduce 闭环：实现、测试、benchmark、NCU | `In Progress` |
| Stage 2 | 补齐 activation、transpose、norm、softmax | `Not Started` |
| Stage 3 | 建立 CUDA Core GEMM baseline | `Not Started` |
| Stage 4 | 实现 WMMA Tensor Core GEMM | `Not Started` |
| Stage 5 | 实现 MMA PTX Tensor Core GEMM | `Not Started` |
| Stage 6 | 引入 CUTLASS GEMM 与 epilogue fusion | `Not Started` |
| Stage 7 | attention / convolution 迁移 Tensor Core 经验 | `Not Started` |

## Rule Of Thumb

- 先补齐常用算子，再追单个算子的极限性能。
- 每个算子至少保留 baseline、优化版、正确性测试、benchmark。
- 对 compute-bound 算子重点看 `TFLOPS`，对 memory-bound 算子重点看 `GB/s`。
- 每次优化尽量留下 profiling 证据，而不是只留下代码。
- Tensor Core 和 CUTLASS 都服务于具体算子，不单独拆成根目录。
