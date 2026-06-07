# Activation and Elementwise Notes

## Scope

- `relu`
- `gelu`
- `silu`
- `sigmoid`
- `exp`
- `add`
- `sub`
- `mul`
- `bias + activation`

## YOLOX Related Operators

- `SiLU / Swish`: activation after convolution blocks.
- `Sigmoid`: used by detection head outputs and decode logic.
- `Exp`: used by bbox decode in some implementations.
- `Add / Sub / Mul`: used by residual paths and bbox decode arithmetic.

## Status

- `Not Started`

## Notes

- Record vectorization, throughput bottlenecks, and fusion strategies for elementwise kernels here.
