# Docs

`docs/` 目录用于记录跨算子、跨平台的学习笔记和问题沉淀。

当前文档：

- `plan.md`：记录项目发展方向、阶段目标和 Tensor Core / CUTLASS 学习计划。
- `questions.md`：记录重要问答与结论。
- `tensor_core.md`：记录 Tensor Core 编程学习路线与关键概念。
- `cutlass.md`：记录 CUTLASS 学习路线、源码阅读和使用方式。
- `cpu_instruction_sets.md`：记录工业常用 CPU SIMD / 矩阵指令集和当前机器支持情况。

原则：

- 算子相关实现细节按需放在对应算子目录的笔记文件中，不强制每个目录都创建 `notes.md`。
- 跨多个算子的通用技术主题放在 `docs/`。
- 第三方库源码不直接放在 `docs/`，真正引入时放到 `third_party/`。
