# 当前模型与消融说明

## 当前用于推理的模型
- 模型结构：DualStream3DCNNLSTM（10 帧序列）
- 权重路径：核心程序\deep_learning\active_learning_results\dualstream_3dcnn_lstm_active.pth
- 加载位置：modern_detection_gui_optimized.py

## 模型结构要点
- 3D CNN：提取时空形态特征
- LSTM：建模统计序列（mean/max/std）随时间的变化
- 融合：拼接形态与统计特征，输出概率/大小/深度

## 消融设计
- Shape Only：仅保留 3D CNN 形态流
- Intensity Only：仅保留统计流
- No LSTM：移除 LSTM，用统计序列均值替代

## 新增 LSTM 消融脚本
- Code_Archive\train_active_dualstream_nolstm.py
- 输出权重：active_learning_results\dualstream_3dcnn_nolstm_base.pth / dualstream_3dcnn_nolstm_active.pth
