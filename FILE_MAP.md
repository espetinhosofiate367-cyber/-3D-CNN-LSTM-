# Refactored Project 文件地图（逐目录/逐文件）

> 目标：让新同学拿到项目后，知道每个目录、每个关键文件是干什么的。

## 顶层目录

- `src/`：可执行代码（核心模型、任务脚本、GUI）
- `logs/`：运行产生的中间与最终输出（模型、指标、图）
- `reports/`：汇总报告和执行日志
- `legacy_mirror/`：历史资产镜像（只读，不作为当前主链入口）

## 顶层文件

- `README.md`：零基础上手手册（从 conda 到启动 GUI）
- `ARCHITECTURE.md`：分层架构说明（资产层/执行层/交付层）
- `environment.yml`：Conda 环境依赖定义
- `.env.example`：环境变量模板
- `.env.runtime`：当前机器的环境变量快照（机器相关）
- `FILE_MAP.md`：本文件，逐文件职责说明

## src/

### src/core/

- `models.py`
  - 定义两个模型结构：
    - `DualStream3DCNNLSTM`（主线）
    - `DualStream3DCNNNoLSTM`（消融）
- `model_factory.py`
  - 统一模型构建与权重命名：
    - `build_model(model_name)`
    - `model_weight_name(model_name)`
- `datasets.py`
  - `ActiveDualStreamDataset`：
    - 读取 CSV（最后 96 列）
    - 10 帧滑窗
    - 归一化 + 统计特征（mean/max/std）
    - 训练/主动学习/验证模式数据组织

### src/tasks/

- `train_active_dualstream.py`
  - 主线模型训练（base + active 两阶段）
  - 输出：
    - `dualstream_3dcnn_lstm_base_refactor.pth`
    - `dualstream_3dcnn_lstm_active_refactor.pth`
    - `train_active_dualstream_refactor_log.json`

- `train_active_nolstm.py`
  - NoLSTM 消融训练（base + active）
  - 输出：
    - `dualstream_3dcnn_nolstm_base_refactor.pth`
    - `dualstream_3dcnn_nolstm_active_refactor.pth`
    - `train_active_nolstm_refactor_log.json`

- `evaluate_file3.py`
  - File3 统一口径评估（dualstream / nolstm）
  - 输出：
    - `file3_metrics_refactor_<model>.json`（帧级 + 文件级汇总）
    - `file3_per_file_metrics_refactor_<model>.json`（文件级明细）

- `compare_models.py`
  - 读取双模型评估 JSON，生成对比图：
    - `comparison_mae_size_depth.png`
    - `comparison_file_level_detection_rate.png`

- `generate_internal_dataflow.py`
  - 生成内部数据流图：
    - `internal_dataflow_upgraded.png`

- `generate_task_report.py`
  - 自动汇总当前产物，生成：
    - `reports/TASK_REPORT.md`

### src/gui/

- `plotly_app.py`
  - Dash 页面入口（端口 8052）
  - 分页：
    - 实时推理
    - 模型指标对比
    - 成果文件分页浏览
    - 图像画廊分页
  - 支持无 `OUTPUT_DIR` 时默认读取 `refactored_project/logs/outputs`

## logs/

### logs/outputs/（运行后自动生成）

常见文件：
- 模型权重（`.pth`）
- 评估结果（`file3_metrics_*.json`, `file3_per_file_metrics_*.json`）
- 训练日志（`train_active_*_log.json`）
- 可视化图（`comparison_*.png`, `internal_dataflow_upgraded.png`）

## reports/

- `TASK_REPORT.md`：自动生成的汇总报告（给师兄看的主报告）
- `DELIVERY_LOG.md`：交付过程日志（含环境、执行动作、阶段结果）

## legacy_mirror/（只读）

- `README.md`：只读说明
- `GitHub_Docs_Package/`：原始文档与代码归档
- `failed_attempts/`：历史尝试与中间产物
- `prestudy/`：前置学习内容
- `references/`：参考文献

> 注意：`legacy_mirror/` 仅用于追溯与比对，不作为当前运行入口。

## 新人建议阅读顺序

1. `README.md`（先跑通）
2. `FILE_MAP.md`（知道每个文件干什么）
3. `ARCHITECTURE.md`（理解整体分层）
4. `reports/TASK_REPORT.md`（看当前结果）
