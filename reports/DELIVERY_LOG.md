# 交付执行日志（2026-02-28）

## 一、目标
按用户要求一次性完成以下事项：
1. 创建项目专用 conda 虚拟环境；
2. 在项目下创建重构目录，迁移有用代码并保持核心思路不变；
3. 完成认领任务（NoLSTM消融 + 评估口径统一）；
4. GUI尽量改为 Plotly 结构；
5. 输出完整可汇报日志。

## 二、已执行动作（可审计）

### 2.1 环境准备
- 检查 conda 可用：`conda 24.11.3`
- 创建环境：`conda create -y -n bigchuang_env python=3.10`
- 安装基础依赖：`numpy pandas scipy matplotlib seaborn plotly dash pyserial tqdm`
- 安装深度学习依赖：`torch torchvision torchaudio (cpu wheel)`

### 2.2 新建重构工程
- 目录：`refactored_project/`
- 文件：
  - `README.md`
  - `environment.yml`
  - `src/core/models.py`
  - `src/core/datasets.py`
  - `src/tasks/train_active_dualstream.py`
  - `src/tasks/train_active_nolstm.py`
  - `src/tasks/evaluate_file3.py`
  - `src/gui/plotly_app.py`
  - `reports/DELIVERY_LOG.md`

## 三、重构说明（保持思路不变）
- 模型思路未变：3D CNN 形态流 + 统计流（LSTM/NoLSTM）+ 三任务头。
- 数据流程未变：10帧滑窗、每帧归一化、统计特征 `mean/max/std`。
- 训练策略未变：基线训练（File1）-> 主动学习混合训练（File1+File2）。
- 评估对象未变：File3 独立测试。

## 四、你认领任务的可执行入口
- 主线训练：
  - `conda run -n bigchuang_env python refactored_project/src/tasks/train_active_dualstream.py`
- NoLSTM训练：
  - `conda run -n bigchuang_env python refactored_project/src/tasks/train_active_nolstm.py`
- 统一口径评估：
  - `conda run -n bigchuang_env python refactored_project/src/tasks/evaluate_file3.py --model dualstream`
  - `conda run -n bigchuang_env python refactored_project/src/tasks/evaluate_file3.py --model nolstm`
- Plotly GUI：
  - `conda run -n bigchuang_env python -m refactored_project.src.gui.plotly_app`

## 五、运行前必须设置的环境变量
```powershell
setx TACTILE_DATA_ROOT "你的建表数据根目录"
setx BASE_LABELS_JSON "你的manual_keyframe_labels.json"
setx FILE2_LABELS_JSON "你的manual_keyframe_labels_file2.json"
setx OUTPUT_DIR "你的输出目录(建议: D:\VScodeProjects\bigchuang\refactored_project\logs\outputs)"
```

## 六、交付给师兄的最小成果包清单
1. `file3_metrics_refactor_dualstream.json`
2. `file3_metrics_refactor_nolstm.json`
3. `train_active_dualstream_refactor_log.json`
4. `train_active_nolstm_refactor_log.json`
5. 一页结论摘要：
   - 哪个模型在统一口径下更优
   - MAE与检出率差异
   - 是否建议保留LSTM

## 七、风险与注意事项
- 当前脚本默认从环境变量读取路径，避免绝对路径硬编码。
- 若使用 GPU，请将 `bigchuang_env` 中 PyTorch 切换为 CUDA 对应版本。
- 评估口径必须固定阈值并双报：帧级 + 文件级，避免口径混淆。

## 八、2026-02-28 实际执行结果补充

### 8.1 GPU环境切换结果
- 设备：NVIDIA GeForce RTX 4060 Laptop GPU
- 驱动/CUDA：Driver 560.94 / CUDA 12.6
- PyTorch：`2.6.0+cu124`
- CUDA可用：`torch.cuda.is_available() == True`

### 8.2 训练执行状态
- DualStream 训练：已完成（BASE 10 + ACTIVE 15）
- NoLSTM 训练：已完成（BASE 10 + ACTIVE 15）
- 训练日志已输出到 `logs/outputs/`

### 8.3 统一口径评估结果（File3）
- DualStream：
  - accuracy: 0.731834
  - f1: 0.481734
  - mae_size: 0.310616
  - mae_depth: 0.739116
- NoLSTM：
  - accuracy: 0.726216
  - f1: 0.423131
  - mae_size: 0.334373
  - mae_depth: 0.698488

### 8.4 Plotly 分页工作台
- 已升级为分页工作台并运行：`http://127.0.0.1:8052`
- 页面：
  - 实时推理
  - 模型指标对比
  - 成果文件分页浏览
  - 图像画廊分页

### 8.5 师兄成果全量镜像迁移
- 已复制到 `refactored_project/legacy_mirror/`：
  - `GitHub_Docs_Package`
  - `failed_attempts`（原“失败尝试”目录）
  - `prestudy`
  - `references`

### 8.6 自动汇报产物
- 自动生成报告：`reports/TASK_REPORT.md`

