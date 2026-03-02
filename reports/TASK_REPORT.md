# 认领任务结果报告（自动生成）

生成时间: 2026-02-28 21:43:21

## 1. 执行范围
- GPU环境切换并验证
- 主线模型训练（DualStream）
- NoLSTM 消融训练
- File3 统一口径双模型评估
- Plotly 分页工作台启动

## 2. 指标对比（File3）
| 指标 | DualStream | NoLSTM |
|---|---:|---:|
| accuracy | 0.731834 | 0.726216 |
| precision | 0.501901 | 0.488627 |
| recall | 0.463125 | 0.373118 |
| f1 | 0.481734 | 0.423131 |
| mae_size | 0.310616 | 0.334373 |
| mae_depth | 0.739116 | 0.698488 |

### 结论建议
- 在当前统一口径下，DualStream 的分类综合表现（F1）优于 NoLSTM，建议保留 LSTM。
- 深度与大小误差请结合业务阈值联合判断，避免只看单一指标。

## 3. 文件级指标（File-level）
| 指标 | DualStream | NoLSTM |
|---|---:|---:|
| file_level_total | 81 | 81 |
| file_level_correct | 30 | 23 |
| file_level_detection_rate | 0.370370 | 0.283951 |
- DualStream 文件级明细条数: 81
- NoLSTM 文件级明细条数: 81

## 4. 训练日志摘要
- DualStream 训练记录条数: 25
- NoLSTM 训练记录条数: 25

## 5. 图表与交付文件
- logs/outputs/comparison_mae_size_depth.png
- logs/outputs/comparison_file_level_detection_rate.png
- logs/outputs/internal_dataflow_upgraded.png
- logs/outputs/file3_metrics_refactor_dualstream.json
- logs/outputs/file3_metrics_refactor_nolstm.json
- logs/outputs/file3_per_file_metrics_refactor_dualstream.json
- logs/outputs/file3_per_file_metrics_refactor_nolstm.json
- logs/outputs/train_active_dualstream_refactor_log.json
- logs/outputs/train_active_nolstm_refactor_log.json

## 6. 300-500字结论（可直接用于汇报）
在同一数据划分（File1/2训练，File3测试）、同一阈值（0.5）和同一统计口径下，本次完成了 DualStream 主线与 NoLSTM 消融的可复现实验对比。从帧级指标看，DualStream 在 accuracy、precision、recall、F1 上均优于 NoLSTM，说明 LSTM 对时序统计流的建模仍提供了稳定增益，尤其在召回与综合判别能力方面更明显。从回归误差看，DualStream 在大小预测 MAE 更优，而 NoLSTM 在深度 MAE 上略有优势，提示两者存在偏差分布差异，后续可通过损失加权或分任务蒸馏进一步平衡。在文件级检出率层面，DualStream 同样保持领先，支持‘保留 LSTM 进入主程序主干’这一工程决策。需要强调的是，所有结论均绑定脚本与输出文件，可在当前环境复跑复核；后续论文写作建议同时报告帧级与文件级口径，避免因指标定义差异引发结论歧义。
