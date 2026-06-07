# CPU 指令集

本文件用于记录工业界常见 CPU SIMD / 矩阵扩展指令集，以及当前开发机器适合优先学习和使用的指令集。

## 为什么需要关注指令集

CPU 高性能算子通常依赖硬件指令集提升吞吐：

- SIMD 指令一次处理多个元素。
- FMA 指令把乘法和加法合并成一条指令。
- 矩阵扩展指令面向 GEMM / Conv / 推理场景提供更高吞吐。

如果 CPU 不支持某个指令集，程序运行到对应指令时可能直接崩溃：

```text
Illegal instruction
```

因此工程上通常需要：

- 编译期为不同指令集生成不同 target。
- 运行期检测 CPU capability。
- 对不支持高级指令集的机器回退到 baseline 实现。

## 工业常用指令集

### x86

| 指令集 | 位宽 | 常见平台 | 典型用途 |
| --- | --- | --- | --- |
| `SSE / SSE2` | 128-bit | 覆盖非常广的老平台 | 基础向量化兼容层 |
| `SSSE3 / SSE4.1 / SSE4.2` | 128-bit | 新老 Intel / AMD CPU | 整数操作、比较、blend、文本/数据处理 |
| `AVX` | 256-bit 浮点 | Intel Sandy Bridge 以后，AMD 视代际而定 | 更宽的浮点向量计算 |
| `AVX2` | 256-bit 浮点 + 整数 | Intel Haswell 以后，现代 AMD | 当前最主流的 CPU kernel 优化目标 |
| `FMA` | 128/256-bit | Intel Haswell 以后，现代 AMD | GEMM、Conv、dot product、多项式近似 |
| `AVX-512` | 512-bit | Intel Xeon、部分桌面/移动平台 | 服务器 GEMM、reduce、norm、quantization |
| `VNNI` | 向量点积风格整数指令 | Intel server/client 部分代际 | INT8 推理 |
| `AMX` | tile 矩阵扩展 | Intel Sapphire Rapids 以后 | BF16 / INT8 GEMM 和推理 |

x86 通用 CPU kernel 的推荐学习顺序：

```text
scalar baseline
AVX2 + FMA
AVX-512
VNNI / AMX
```

### ARM

| 指令集 | 位宽 | 常见平台 | 典型用途 |
| --- | --- | --- | --- |
| `NEON` | 128-bit | 移动端、嵌入式、Apple/ARM 平台 | ARM 最主流 SIMD |
| `SVE` | 可变向量长度 | ARM server / HPC | 向量长度无关的 SIMD |
| `SVE2` | 可变向量长度 | 较新的 ARM 平台 | 更完整的整数和通用向量能力 |
| `SME` | 矩阵扩展 | 较新的 ARM 架构 | 矩阵/tile 加速 |

ARM 推荐学习顺序：

```text
NEON
SVE / SVE2
SME
```

### RISC-V

| 指令集 | 特点 | 典型用途 |
| --- | --- | --- |
| `RVV` | 可变长度向量扩展 | RISC-V 平台上的可移植向量 kernel |

RISC-V 向量扩展很有潜力，但从当前工业算子工程岗位看，生态普及程度仍低于 x86 的 `AVX2 / AVX-512` 和 ARM 的 `NEON / SVE`。

## 当前机器

当前开发环境检测到：

```text
Model name: 13th Gen Intel(R) Core(TM) i7-13650HX
```

你提到的是 `i7-13650H`。从 SIMD 学习和本机优化角度看，`i7-13650H` 和当前环境检测到的 `i7-13650HX` 给出的实践建议一致：

```text
优先使用 AVX2 + FMA 作为本机 CPU 优化目标。
```

当前环境检测到的相关 flags：

```text
sse4_1
sse4_2
avx
avx2
fma
```

当前未检测到：

```text
avx512
amx
```

## 本机推荐路线

在当前 Intel 13 代移动端 CPU 上，本项目推荐：

1. 先写 scalar baseline。
2. 再写 AVX2 + FMA 优化版本。
3. 加入 cache blocking / tiling。
4. 在 GEMM / Conv 这类算子中加入 packing。
5. 单线程 kernel 稳定后，再加入多线程。
6. `AVX-512 / AMX` 先作为服务器侧知识点，不作为本机默认目标。

常用编译参数：

```bash
g++ kernel.cpp -O3 -mavx2 -mfma -o kernel
```

CMake target 示例：

```cmake
target_compile_options(your_target PRIVATE -O3 -mavx2 -mfma)
```

学习阶段也可以临时使用：

```bash
-march=native
```

但不要在需要跨机器分发的二进制里依赖 `-march=native`，因为它可能生成其它机器不支持的指令。

## 常用检查命令

查看 CPU 型号和 flags：

```bash
lscpu
```

只查看 SIMD 相关 flags：

```bash
grep -m1 -o 'sse4_1\|sse4_2\|avx\|avx2\|fma\|avx512[^ ]*\|amx[^ ]*' /proc/cpuinfo | sort -u
```

## 算子优化关注点

| 算子 | 相关 CPU 优化能力 |
| --- | --- |
| `activation / elementwise` | AVX2 向量数学、近似函数 |
| `reduce` | AVX2 horizontal reduction、多线程 partial reduce |
| `norm` | AVX2 reduce + elementwise fusion |
| `gemv` | AVX2 FMA、cache locality |
| `gemm` | AVX2 FMA、blocking、packing、register micro-kernel |
| `convolution` | im2col + GEMM、direct conv、layout transform |
| `quantization` | AVX2 整数操作，如果支持则关注 VNNI |
| `transpose` | cache blocking、向量化 load/store |
