# Benchmarks

`benchmarks/` 用于存放性能测试与对标脚本。

这里不负责数值正确性验证，而是专注：

- latency
- throughput
- TFLOPS / GB/s
- speedup
- 与工业标准实现的对比

约定：

- `*_benchmark.cu` 是给稳定 latency 测量和 Nsight Compute profiling 使用的 C++/CUDA driver。
- `bench_*_torch.py` 是参考实现 benchmark，用于和 PyTorch 等工业实现做性能对比。
- 算子数值正确性测试放在 `tests/`，不放在这里。
