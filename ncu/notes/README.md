# NCU Notes

这里用于记录学习 Nsight Compute **工具本身**的使用心得，例如：

- 如何判断 kernel 是 compute-bound 还是 memory-bound
- 哪类 warp stall 最突出、怎么读
- occupancy 受寄存器还是 shared memory 限制，怎么从报告里看出来
- 常用命令与 section 的取舍

具体某个算子的三层性能分析结果不放这里，而是放在该算子自己的目录下，例如
`tests/cuda/test_reduce/profiling/`。

