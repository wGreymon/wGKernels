# Convolution Notes

## Scope

- `conv2d`
- `depthwise conv`
- `im2col + gemm`

## YOLOX Related Operators

- `Conv2d`: used throughout backbone, neck, and head.
- `Depthwise Conv2d`: used by depthwise variants.

## Status

- `Not Started`

## Notes

- Record direct convolution, implicit GEMM, and layout-dependent optimization strategies here.
