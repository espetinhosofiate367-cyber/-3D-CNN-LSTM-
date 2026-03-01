# legacy_mirror 使用说明（只读）

该目录用于保留师兄原始成果与历史快照，供追溯与对照使用。

## 约束
- 只读使用：不要在该目录直接运行训练、评估或推理脚本。
- 不作为当前主链入口：当前可执行主链以 `refactored_project/src` 为准。
- 允许比对：可用于论文复现对照、结果核查与历史行为解释。

## 推荐做法
- 训练与评估：使用 `refactored_project/src/tasks` 下脚本。
- 可视化与交互：在项目根目录执行 `conda run -n bigchuang_env python -m refactored_project.src.gui.plotly_app`。
- 输出产物：统一写入 `refactored_project/logs/outputs`。
