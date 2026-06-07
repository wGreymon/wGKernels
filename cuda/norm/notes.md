# Norm Notes

## Scope

- `layernorm`
- `rmsnorm`
- `groupnorm`
- `batchnorm2d`

## YOLOX Related Operators

- `BatchNorm2d`: used after convolution blocks; inference mode can be implemented as per-channel affine transform.

## Status

- `Not Started`

## Notes

- Record mean/variance computation strategy, vectorized loads, and fusion opportunities here.
