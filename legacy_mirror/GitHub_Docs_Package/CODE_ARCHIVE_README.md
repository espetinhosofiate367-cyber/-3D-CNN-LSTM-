# 代码归档说明

## 归档目录
- GitHub_Docs_Package\Code_Archive

## 文件作用说明
- modern_detection_gui_optimized.py：主程序 GUI，集成串口采集与 DL 推理
- realtime_detector.py：实时检测器与推理封装
- dualstream_3dcnn_lstm.py：双流 3D CNN + LSTM 与无 LSTM 结构
- train_active_cnn3d_lstm.py：3D CNN→LSTM 新方案训练脚本
- active_dualstream_dataset.py：主动学习序列数据集与统计序列构造
- train_active_dualstream.py：主动学习训练脚本（基准）
- train_active_dualstream_nolstm.py：移除 LSTM 的消融训练脚本
- evaluate_file3_cnn3d_lstm.py：CNN3D→LSTM 的 File3 评估
- compare_ablation_models.py：多模型消融对比图
- generate_dataflow_diagram.py：输入输出数据流程图
- evaluate_file3_active_model.py：File3 评估与指标统计
- compare_file123_similarity.py：三文件对比与一致性评估
- generate_candidate_segments.py：难例候选生成
- inference.py：离线推理与批量评估
- main_gui.py：Release 版本主程序入口
- fusion_real_time_detection.py：轻量级实时检测模块
- final_model.py：Release 版本模型定义
- sequence_dataset.py：数据加载与预处理
- train_ablation.py：消融训练脚本
- plot_ablation_results.py：消融结果绘图
- calculate_detection_rate.py：检测率统计
