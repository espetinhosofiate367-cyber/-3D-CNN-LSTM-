import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dualstream_3dcnn_lstm import DualStream3DCNNLSTM

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_csv_data(path):
    df = pd.read_csv(path)
    mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
    if mat_cols:
        mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
        data = df[mat_cols].values.astype(np.float32)
    else:
        data = df.iloc[:, -96:].values.astype(np.float32)
    return data


def normalize_sequence(seq_raw):
    seq_len = len(seq_raw)
    seq_out = np.zeros((seq_len, 1, 12, 8), dtype=np.float32)
    for i in range(seq_len):
        frame = seq_raw[i]
        d_min, d_max = frame.min(), frame.max()
        if d_max - d_min > 1e-6:
            frame = (frame - d_min) / (d_max - d_min)
        else:
            frame = frame - d_min
        seq_out[i, 0] = frame.reshape(12, 8)
    return seq_out


def stats_sequence(seq_raw):
    seq_len = len(seq_raw)
    stats = np.zeros((seq_len, 3), dtype=np.float32)
    for i in range(seq_len):
        fr = seq_raw[i]
        stats[i] = np.array([np.mean(fr), np.max(fr), np.std(fr)], dtype=np.float32)
    return stats


def pick_sample(labels_map, data_root, seq_len=10):
    for rel_path, info in labels_map.items():
        segments = info.get('segments', [])
        if not segments:
            continue
        safe_rel = rel_path.replace('\\', os.sep).replace('/', os.sep)
        base_dir = os.path.dirname(safe_rel)
        file_path = os.path.join(data_root, base_dir, '1.CSV')
        if not os.path.exists(file_path):
            continue
        data = read_csv_data(file_path)
        for start, end in segments:
            if end - start < seq_len:
                continue
            end_row = start + seq_len - 1
            if end_row >= len(data):
                continue
            return file_path, data, end_row
    return None, None, None


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\数据集 最新\建表数据'
    labels_path = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\manual_keyframe_labels.json'
    model_path = r'c:\Users\SWH\Desktop\智能医疗检测系统\核心程序\deep_learning\active_learning_results\dualstream_3dcnn_lstm_active.pth'
    output_dir = os.path.join(base_dir, 'active_learning_results')
    os.makedirs(output_dir, exist_ok=True)

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_map = json.load(f)

    file_path, data, end_row = pick_sample(labels_map, data_root, seq_len=10)
    if file_path is None:
        print('No valid sample found')
        return

    seq_raw = data[end_row - 9: end_row + 1]
    seq_norm = normalize_sequence(seq_raw)
    stats = stats_sequence(seq_raw)

    raw_last = seq_raw[-1].reshape(12, 8)
    norm_last = seq_norm[-1, 0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStream3DCNNLSTM(seq_len=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.tensor(seq_norm).unsqueeze(0).to(device)
    x3d = x.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        x_shape = torch.relu(model.bn1(model.conv1(x3d)))
        x_shape = model.pool1(x_shape)
        x_shape = torch.relu(model.bn2(model.conv2(x_shape)))
        x_shape = model.pool2(x_shape)
        b, c, t, h, w = x_shape.shape
        shape_map = x_shape.mean(dim=(1, 2)).squeeze(0).cpu().numpy()

        stats_tensor = torch.tensor(stats).unsqueeze(0).to(device)
        stats_embed = model.stats_fc(stats_tensor)
        lstm_out, _ = model.lstm(stats_embed)
        lstm_energy = torch.mean(torch.abs(lstm_out), dim=2).squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ax = axes[0, 0]
    im0 = ax.imshow(raw_last, cmap='viridis')
    ax.set_title('原始末帧 (12x8)')
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im1 = ax.imshow(norm_last, cmap='viridis')
    ax.set_title('归一化末帧')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    im2 = ax.imshow(shape_map, cmap='magma')
    ax.set_title('3D CNN 形态特征(均值)')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    ax.plot(stats[:, 0], label='Mean')
    ax.plot(stats[:, 1], label='Max')
    ax.plot(stats[:, 2], label='Std')
    ax.set_title('统计特征序列')
    ax.set_xlabel('帧索引')
    ax.legend()

    ax = axes[1, 1]
    ax.plot(lstm_energy)
    ax.set_title('LSTM 序列响应强度')
    ax.set_xlabel('帧索引')

    ax = axes[1, 2]
    ax.axis('off')
    ax.text(0, 0.8, f'样本路径:\n{file_path}', fontsize=8)
    ax.text(0, 0.55, f'end_row: {end_row}', fontsize=10)
    ax.text(0, 0.35, '序列长度: 10 帧', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'original_pipeline_steps.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(out_path)


if __name__ == '__main__':
    main()
