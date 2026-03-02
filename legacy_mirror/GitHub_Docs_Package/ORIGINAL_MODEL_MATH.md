# 原模型数学细节（含公式）

## 1. 输入与符号约定
- 触觉序列：\(\mathbf{X} = \{\mathbf{x}_t\}_{t=1}^T\), \(T=10\)
- 单帧阵列：\(\mathbf{x}_t \in \mathbb{R}^{12\times 8}\)
- 归一化后输入：\(\tilde{\mathbf{x}}_t\)
- 统计序列：\(\mathbf{s}_t = [\mu_t, m_t, \sigma_t]^\top\in\mathbb{R}^3\)

## 2. 逐帧归一化
对每帧做 min-max 归一化：

\[
\tilde{\mathbf{x}}_t = \begin{cases}
\dfrac{\mathbf{x}_t - \min(\mathbf{x}_t)}{\max(\mathbf{x}_t) - \min(\mathbf{x}_t)}, & \max \neq \min \\
\mathbf{x}_t - \min(\mathbf{x}_t), & \max = \min
\end{cases}
\]

## 3. 统计特征序列

\[
\mu_t = \dfrac{1}{96}\sum_{i=1}^{96} x_{t,i},\quad
m_t = \max_i x_{t,i},\quad
\sigma_t = \sqrt{\dfrac{1}{96}\sum_{i=1}^{96}(x_{t,i}-\mu_t)^2}
\]

于是：\(\mathbf{s}_t = [\mu_t, m_t, \sigma_t]^\top\)。

## 4. 形态流（3D CNN）
将序列堆叠为：\(\tilde{\mathbf{X}}\in\mathbb{R}^{T\times 1\times 12\times 8}\)。
使用 3D 卷积与池化提取时空特征：

\[
\mathbf{F}_1 = \mathrm{Pool}_1\big(\mathrm{ReLU}(\mathrm{BN}(\mathbf{W}_1 * \tilde{\mathbf{X}}))\big)
\]

\[
\mathbf{F}_2 = \mathrm{Pool}_2\big(\mathrm{ReLU}(\mathrm{BN}(\mathbf{W}_2 * \mathbf{F}_1))\big)
\]

将时间维与通道折叠后进行 2D 卷积与全局池化：

\[
\mathbf{H} = \mathrm{Conv2D}(\mathrm{reshape}(\mathbf{F}_2))
\]

\[
\mathbf{z}_\text{shape} = \mathrm{GAP}(\mathbf{H})\in\mathbb{R}^{32}
\]

## 5. 统计流（LSTM）
先将统计向量嵌入：

\[
\mathbf{e}_t = \phi(\mathbf{W}_s\mathbf{s}_t + \mathbf{b}_s),\quad \mathbf{e}_t\in\mathbb{R}^{16}
\]

双向 LSTM 聚合：

\[
\mathbf{h}_t = \mathrm{BiLSTM}(\mathbf{e}_t)
\]

序列级特征采用时间平均：

\[
\mathbf{z}_\text{stat} = \dfrac{1}{T}\sum_{t=1}^{T}\mathbf{h}_t \in\mathbb{R}^{2H}
\]

其中 \(H=64\)。

## 6. 融合与输出
拼接融合：

\[
\mathbf{z} = [\mathbf{z}_\text{shape};\mathbf{z}_\text{stat}]
\]

\[
\mathbf{f} = \phi(\mathbf{W}_f\mathbf{z} + \mathbf{b}_f)
\]

三任务输出：

\[
\hat{y}_p = \mathbf{W}_p\mathbf{f} + \mathbf{b}_p
\]

\[
\hat{y}_s = \mathbf{W}_s'\mathbf{f} + \mathbf{b}_s'
\]

\[
\hat{y}_d = \mathbf{W}_d'\mathbf{f} + \mathbf{b}_d'
\]

概率为：\(p = \sigma(\hat{y}_p)\)。

## 7. 损失函数
分类损失：

\[
\mathcal{L}_{cls} = \mathrm{BCEWithLogits}(\hat{y}_p, y_p)
\]

回归仅在预测为阳性时计算：

\[
\mathcal{M} = \mathbb{I}[\sigma(\hat{y}_p) > 0.5]
\]

\[
\mathcal{L}_{size} = \mathcal{M}\cdot \|\hat{y}_s - y_s\|_2^2,\quad
\mathcal{L}_{depth} = \mathcal{M}\cdot \|\hat{y}_d - y_d\|_2^2
\]

总损失：

\[
\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{size} + \mathcal{L}_{depth}
\]

## 8. 代码对应关系
- 归一化与统计：
  - evaluate_file3_active_model.py / sequence_dataset.py
- 模型结构：
  - Code_Archive\dualstream_3dcnn_lstm.py
- 损失实现：
  - Code_Archive\train_active_dualstream.py
