# CNN3D→LSTM 方案输入输出流程报告

## 流程概述
1. 传感器/CSV 读取 96 维触觉阵列帧
2. 滑窗构建 10 帧序列
3. 逐帧归一化 + 统计特征计算
4. 3D CNN 提取形态特征（保留时间维）
5. LSTM 在时间维上聚合，得到序列级特征
6. 多任务头输出：异常概率 + 结节大小 + 深度

## 图示
- Code_Archive\active_learning_results\dataflow_cnn3d_lstm.png

## 训练与评估脚本
- 训练：Code_Archive\train_active_cnn3d_lstm.py
- 评估：Code_Archive\evaluate_file3_cnn3d_lstm.py
- 对比：Code_Archive\compare_ablation_models.py
