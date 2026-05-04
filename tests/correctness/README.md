# Correctness Tests

正确性测试与性能 benchmark 分离维护。

这里的目标是验证算子输出是否正确，当前默认参考实现为 `PyTorch`，后续按算子类别补充更细的 shape、dtype 和容差策略。
