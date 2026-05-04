# NCU Metric Sets

这里用于整理不同类型 kernel 的常用分析指标集合。

建议后续至少维护两类：

- `memory_bound`
- `compute_bound`

例如：

- reduce / transpose 更关注带宽、访存合并与 stall
- gemm / attention 更关注 tensor core 利用率、occupancy 与指令吞吐
