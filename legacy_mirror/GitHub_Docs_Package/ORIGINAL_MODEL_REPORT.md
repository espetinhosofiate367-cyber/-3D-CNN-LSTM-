# 原模型（DualStream3DCNNLSTM）架构与数据流

## 架构概览
原模型为双流结构：
- 形态流：3D CNN 从 10 帧触觉序列中提取时空形态特征
- 统计流：LSTM 对每帧统计特征（mean/max/std）序列建模
- 融合：拼接后进入全连接层，输出概率/大小/深度

## 输入输出
- 输入：10 帧序列（12x8 触觉阵列）
- 输出：
  - 结节存在概率（logits + sigmoid）
  - 结节大小（cm）
  - 结节深度（cm）

## 数据流步骤
1. CSV 读取原始 96 维阵列帧
2. 滑窗构建 10 帧序列
3. 逐帧归一化，形成 (10,1,12,8)
4. 统计序列：每帧 mean/max/std → (10,3)
5. 形态流：3D CNN → 时空形态特征
6. 统计流：LSTM → 序列聚合特征
7. 融合 → 概率/大小/深度

## 可视化
- 原模型中间步骤可视化图：
  GitHub_Docs_Package\Code_Archive\active_learning_results\original_pipeline_steps.png

## 关键代码
- 模型结构：Code_Archive\dualstream_3dcnn_lstm.py
- 训练脚本：Code_Archive\train_active_dualstream.py
- 推理入口：核心程序\modern_detection_gui_optimized.py
