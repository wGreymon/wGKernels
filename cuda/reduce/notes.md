# Reduce Notes

## Scope

- `sum`
- `max`
- `argmax`
- `softmax`
- `logsumexp`

## Status

- `In Progress`

## Implemented

- `sum`
- `max`
- `argmax`

## Pending

- `softmax`
- `logsumexp`

## Notes

- Current implementation targets 1D `float32` reduction on CUDA.
- Host API is exposed via `include/cuda/reduce.hpp`.
- Each concrete operator has its own implementation file:
  - `cuda/reduce/src/sum.cu`
  - `cuda/reduce/src/max.cu`
  - `cuda/reduce/src/argmax.cu`
- Shared reduce utilities live in `utils/reduce_utils.cuh`.
- Kernel versions are kept in the corresponding operator file.
- `sum` and `max` currently default to `v2` shuffle-based kernels.
- `argmax` currently uses the `v1` shared-memory block reduction path.
- Current benchmark and correctness entry points:
  - `PYTHONPATH=build/python python3 tests/cuda/test_reduce/scripts/test_reduce_vs_pytorch.py`
  - `./build/benchmarks/cuda/wgkernel_reduce_benchmark --op sum --numel 16777216 --warmup 10 --repeat 100`
  - `python3 benchmarks/cuda/bench_reduce_torch.py --op sum --numel 16777216 --warmup 10 --repeat 100`

## Current Design

- First stage uses grid-stride accumulation.
- `v1` block-level reduction is done through shared memory.
- `v2` uses warp-level shuffle first, then shared memory only for cross-warp results.
- Multi-stage reduction uses ping-pong workspace buffers.
- `argmax` uses `(value, index)` pairs and keeps the first index on ties.

## Current Benchmark Snapshot

Measured on the current local NVIDIA CUDA environment:

| Op | wGKernel latency (ms) | PyTorch latency (ms) |
| --- | --- | --- |
| `sum` | `0.2765` | `0.2844` |
| `max` | `0.2766` | `0.2853` |
| `argmax` | `0.2937` | `0.3102` |

## Next Optimization Directions

- Add a shuffle-based `v2` path for `argmax`.
- Add vectorized loads and more aggressive unrolling.
- Introduce a reusable workspace / runner utility to reduce boilerplate in tests and benchmarks.
- Extend from scalar reduction to row-wise / axis-wise reduction.
