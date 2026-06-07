# Resize Notes

## Scope

- nearest-neighbor upsample
- bilinear upsample
- resize with scale factor

## YOLOX Related Operators

- `Upsample`: used in the FPN top-down path to enlarge high-level feature maps before concatenation with shallower features.

## Status

- `Not Started`

## Notes

- Record coordinate mapping, interpolation mode, memory access pattern, and PyTorch comparison results here.
