# Attention Notes

## Scope

- `qk matmul`
- `masked softmax`
- `attention`
- `flash attention`
- `linear attention`

## Status

- `Not Started`

## Notes

- Current placeholder files:
  - `cuda/attention/src/standard_attention.cu`
  - `cuda/attention/src/linear_attention.cu`
  - `cuda/attention/src/flash_attention_like.cu`
- Record data layout, online softmax, shared-memory usage, and prefill/decode optimization ideas here.
- For `linear attention`, record feature-map choice, causal prefix accumulation strategy, normalization denominator, and numerical stability considerations.
