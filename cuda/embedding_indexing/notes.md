# Embedding and Indexing Notes

## Scope

- `embedding`
- `gather`
- `scatter`
- `slice`
- `strided slice`
- `topk`
- `sort`

## YOLOX Related Operators

- `Slice / Strided Slice`: used by Focus-like spatial-to-channel rearrangement in some YOLO variants.
- `Gather`: used by postprocess or index-based selection.
- `TopK / Sort`: used by candidate filtering in some postprocess pipelines.

## Status

- `Not Started`

## Notes

- Record irregular memory access handling, cache behavior, and index layout considerations here.
