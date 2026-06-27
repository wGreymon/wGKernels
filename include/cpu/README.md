# CPU Headers

`include/cpu/` 声明 CPU 朴素 baseline 的对外接口。

约定：

- CPU 实现优先保证简单可信。
- SIMD/指令集加速接口放在 `include/simd/`。
- 具体实现放在 `cpu/<op>/` 下。
