# NCU Scripts

这里用于存放 `ncu` 分析脚本。

示例用途：

- `profile_reduce.sh`
- `profile_gemm.sh`
- `profile_attention.sh`

建议脚本固定输入 shape、dtype 和重复次数，保证不同版本 kernel 的分析结果可复现。
