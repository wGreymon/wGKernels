# Transpose and Layout Transform Notes

## Scope

- `transpose`
- `permute`
- `nchw <-> nhwc`
- `concat`
- `reshape`
- `view`

## YOLOX Related Operators

- `Concat`: used by CSP, FPN, and PAN feature merging.
- `Reshape / View`: used by detection head output formatting; often metadata-only when contiguous layout is compatible.
- `Permute`: used by output layout conversion and decode preparation.

## Status

- `Not Started`

## Notes

- Record coalesced access, shared-memory tiling, and bank-conflict avoidance strategies here.
