# Refactored Tactile Nodule Project（零基础使用手册）

这份文档面向**第一次接触该项目**的同学。你只要按顺序执行，就可以从 0 开始完成：
1. 安装并配置 Conda 环境
2. 跑通训练与评估
3. 打开 Plotly 页面查看结果

---

## 1. 这个项目是做什么的？

该目录是师兄原项目的重构版，核心思路不变：
- 模型：DualStream3DCNNLSTM（主线）+ NoLSTM（消融）
- 训练：主动学习两阶段
- 评估：File3 统一口径（帧级 + 文件级）
- 展示：Plotly/Dash 可视化页面

---

## 2. 目录结构（先知道去哪找东西）

- `src/core/models.py`：模型定义
- `src/core/model_factory.py`：统一模型入口（推荐）
- `src/core/datasets.py`：数据集构建
- `src/tasks/train_active_dualstream.py`：主线训练
- `src/tasks/train_active_nolstm.py`：NoLSTM 训练
- `src/tasks/evaluate_file3.py`：评估脚本
- `src/tasks/compare_models.py`：生成对比图
- `src/tasks/generate_task_report.py`：生成汇总报告
- `src/gui/plotly_app.py`：可视化界面
- `logs/outputs/`：模型权重、评估 JSON、图像产物
- `reports/TASK_REPORT.md`：自动汇总报告

### 2.1 配置与说明文件（重要）

- `environment.yml`：Conda 环境定义（首次建环境就看它）
- `.env.example`：环境变量模板（复制后改成你自己的路径）
- `.env.runtime`：当前机器的实际运行变量快照（机器相关，不建议直接复制给别人）
- `ARCHITECTURE.md`：项目分层架构说明
- `FILE_MAP.md`：逐目录、逐文件用途说明（建议新人先看）

---

## 3. 第一次配置 Conda（Windows）

### 3.1 打开终端并进入项目根目录

在 PowerShell 执行：

```powershell
cd D:\VScodeProjects\bigchuang
```

### 3.2 创建环境（首次仅需一次）

```powershell
conda env create -f refactored_project/environment.yml
```

如果环境已存在，可跳过创建，直接激活。

### 3.3 激活环境

```powershell
conda activate bigchuang_env
```

### 3.4 验证 Python 与关键包

```powershell
python -V
python -c "import torch, dash, plotly; print('ok')"
```

---

## 4. 配置数据路径（必须）

本项目依赖环境变量定位数据与输出目录。直接复制下面命令：

```powershell
$dataRoot = 'D:\VScodeProjects\bigchuang\GitHub_Docs_Package\整理好的数据集\建表数据'
$baseLabels = 'D:\VScodeProjects\bigchuang\GitHub_Docs_Package\manual_keyframe_labels.json'
$file2Labels = 'D:\VScodeProjects\bigchuang\GitHub_Docs_Package\整理好的数据集\manual_keyframe_labels_file2.json'
$outDir = 'D:\VScodeProjects\bigchuang\refactored_project\logs\outputs'

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$env:TACTILE_DATA_ROOT = $dataRoot
$env:BASE_LABELS_JSON = $baseLabels
$env:FILE2_LABELS_JSON = $file2Labels
$env:OUTPUT_DIR = $outDir
$env:PYTHONPATH = 'D:\VScodeProjects\bigchuang'
```

> 说明：以上是“当前终端会话有效”。开新终端后请重新执行，或写入你自己的启动脚本。

如果你想把配置保存为文件，可执行：

```powershell
Copy-Item refactored_project/.env.example refactored_project/.env.runtime
```

然后打开 `refactored_project/.env.runtime`，把路径改成你自己机器上的真实路径。

---

## 5. 最小可运行流程（建议按顺序）

### 5.1 训练主线模型

```powershell
python refactored_project/src/tasks/train_active_dualstream.py
```

### 5.2 训练消融模型（NoLSTM）

```powershell
python refactored_project/src/tasks/train_active_nolstm.py
```

### 5.3 评估（双模型）

```powershell
python refactored_project/src/tasks/evaluate_file3.py --model dualstream
python refactored_project/src/tasks/evaluate_file3.py --model nolstm
```

### 5.4 生成对比图与报告

```powershell
python refactored_project/src/tasks/compare_models.py
python refactored_project/src/tasks/generate_task_report.py
```

完成后重点看：
- `refactored_project/reports/TASK_REPORT.md`
- `refactored_project/logs/outputs/`

---

## 6. 启动 Plotly 页面

在项目根目录执行：

```powershell
conda run -n bigchuang_env python -m refactored_project.src.gui.plotly_app
```

浏览器打开：
- `http://127.0.0.1:8052/`

> 终端会一直占用，这是正常的 Web 服务运行状态。不要关闭该终端。

---

## 7. Plotly 服务重启（页面打不开时）

### 7.1 停掉旧进程（8052 端口）

```powershell
$p=(Get-NetTCPConnection -LocalPort 8052 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)
if($p){Stop-Process -Id $p -Force}
```

### 7.2 重新启动

```powershell
conda run -n bigchuang_env python -m refactored_project.src.gui.plotly_app
```

---

## 8. 常见报错与处理

### 报错 1：`ModuleNotFoundError: No module named 'refactored_project'`

处理：
- 确认当前目录是 `D:\VScodeProjects\bigchuang`
- 使用 `python -m refactored_project.src.gui.plotly_app` 形式启动（不要随意改成其他路径）

### 报错 2：页面显示“尚未找到评估结果”

处理：
- 先运行第 5.3 的两个评估命令
- 检查 `refactored_project/logs/outputs/` 下是否有 `file3_metrics_refactor_*.json`

### 报错 3：`Address already in use` 或页面打不开

处理：
- 按第 7 节先杀端口 8052 再重启

### 报错 4：运行很慢或显存不足

处理：
- 先只跑评估和 GUI 验证流程，训练放在空闲时执行
- 确认 GPU 驱动与 PyTorch 版本匹配

---

## 9. 给师兄的快速验收路径（30 秒）

1. 看 `refactored_project/reports/TASK_REPORT.md`（结论与指标）
2. 看 `refactored_project/logs/outputs/`（权重、JSON、图表）
3. 启动 Plotly 并查看 4 个分页

如果要了解每个目录和每个脚本具体做什么，直接看：`refactored_project/FILE_MAP.md`

---

## 10. 一条龙命令（复制后逐行执行）

```powershell
cd D:\VScodeProjects\bigchuang
conda activate bigchuang_env

$env:TACTILE_DATA_ROOT = 'D:\VScodeProjects\bigchuang\GitHub_Docs_Package\整理好的数据集\建表数据'
$env:BASE_LABELS_JSON = 'D:\VScodeProjects\bigchuang\GitHub_Docs_Package\manual_keyframe_labels.json'
$env:FILE2_LABELS_JSON = 'D:\VScodeProjects\bigchuang\GitHub_Docs_Package\整理好的数据集\manual_keyframe_labels_file2.json'
$env:OUTPUT_DIR = 'D:\VScodeProjects\bigchuang\refactored_project\logs\outputs'
$env:PYTHONPATH = 'D:\VScodeProjects\bigchuang'

python refactored_project/src/tasks/evaluate_file3.py --model dualstream
python refactored_project/src/tasks/evaluate_file3.py --model nolstm
python refactored_project/src/tasks/generate_task_report.py

conda run -n bigchuang_env python -m refactored_project.src.gui.plotly_app
```
