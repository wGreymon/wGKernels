# CUTLASS Backend

`cutlass/` 存放基于 CUTLASS 的算子实现。

定位：

- 不手写 CUDA kernel。
- 调用 CUTLASS 组件、模板或示例实现算子。
- 作为 `cuda/` 手写实现的工业级性能与设计参考。

当前阶段先建立目录骨架，具体算子逐步补齐。
