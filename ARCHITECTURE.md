# Refactored Project Architecture

## 设计目标
- 完整保留师兄已有研究资产（代码/文档/结果）
- 核心算法重构为可维护模块（训练、评估、GUI 解耦）
- 形成“研究资产 + 运行系统 + 汇报产物”三层闭环

## 目录分层

## 1) `legacy_mirror/`（原成果镜像层）
- `GitHub_Docs_Package/`：文档、Code_Archive、结果图、整理数据集
- `failed_attempts/`：历史尝试与主动学习中间产物
- `prestudy/`：前置学习材料
- `references/`：参考文献 PDF

定位：**只读资产层**，用于追溯、复现和成果展示。

## 2) `src/`（重构执行层）
- `core/`：模型与数据模块
- `tasks/`：训练、评估、汇总脚本
- `gui/`：Plotly/Dash 工作台

定位：**可演进工程层**，用于你认领任务的持续实现。

## 3) `logs/` 与 `reports/`（交付层）
- `logs/outputs/`：权重、metrics、训练日志
- `reports/`：可直接给师兄的执行日志与结论报告

定位：**汇报与审计层**，支持科研交付和复盘。

## Plotly 工作台（分页）
- 实时推理页：上传 CSV，逐帧热图 + 概率/大小/深度
- 模型指标对比页：DualStream vs NoLSTM 统一口径对比
- 成果文件分页页：浏览 legacy_mirror 全量文件清单
- 图像画廊分页页：浏览历史可视化图与新评估图
