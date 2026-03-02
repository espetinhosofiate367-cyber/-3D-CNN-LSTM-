import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from dualstream_3dcnn_lstm import DualStream3DCNNLSTM


def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_csv_data(path):
    import pandas as pd
    df = pd.read_csv(path)
    mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
    if mat_cols:
        mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
        data = df[mat_cols].values.astype(np.float32)
    else:
        data = df.iloc[:, -96:].values.astype(np.float32)
    return data


def build_label_array(num_frames, segments):
    labels = np.zeros(num_frames, dtype=np.int32)
    for start, end in segments:
        start = max(0, int(start))
        end = min(num_frames, int(end))
        labels[start:end] = 1
    return labels


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


def align_to_center(img):
    h, w = img.shape
    max_loc = np.unravel_index(np.argmax(img), img.shape)
    center_h, center_w = h // 2, w // 2
    shift_h = center_h - max_loc[0]
    shift_w = center_w - max_loc[1]
    new_img = np.full_like(img, img.min())
    for r in range(h):
        for c in range(w):
            new_r = r + shift_h
            new_c = c + shift_w
            if 0 <= new_r < h and 0 <= new_c < w:
                new_img[new_r, new_c] = img[r, c]
    return new_img


def evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\数据集 最新\建表数据'
    labels_path = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\manual_keyframe_labels.json'
    model_path = os.path.join(base_dir, 'active_learning_results', 'dualstream_3dcnn_lstm_active.pth')
    output_dir = os.path.join(base_dir, 'active_learning_results', 'file3_eval')
    os.makedirs(output_dir, exist_ok=True)

    labels_map = load_labels(labels_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStream3DCNNLSTM(seq_len=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tp = tn = fp = fn = 0
    size_errors = []
    depth_errors = []
    correct_within_step = 0
    total_positive = 0
    best_samples = {}
    best_per_file = {}
    step_size = 0.25
    step_depth = 0.5

    for rel_path, info in labels_map.items():
        try:
            size_val = float(info['size'].replace('cm大', '').replace('cm', ''))
            depth_val = float(info['depth'].replace('cm深', '').replace('cm', ''))
        except Exception:
            continue

        safe_rel = rel_path.replace('\\\\', os.sep).replace('\\', os.sep).replace('/', os.sep)
        base_dir_rel = os.path.dirname(safe_rel)
        file3_path = os.path.join(data_root, base_dir_rel, '3.CSV')
        if not os.path.exists(file3_path):
            continue

        data = read_csv_data(file3_path)
        num_frames = len(data)
        labels = build_label_array(num_frames, info.get('segments', []))

        for end_row in range(9, num_frames):
            seq_raw = data[end_row - 9: end_row + 1]
            if np.isnan(seq_raw).any() or np.all(seq_raw == 0):
                continue

            seq_norm = normalize_sequence(seq_raw)
            stats = stats_sequence(seq_raw)

            input_tensor = torch.tensor(seq_norm).unsqueeze(0).to(device)
            stats_tensor = torch.tensor(stats).unsqueeze(0).to(device)

            with torch.no_grad():
                prob, p_size, p_depth = model(input_tensor, stats_tensor)
            prob_val = float(torch.sigmoid(prob).item())
            pred = 1 if prob_val >= 0.5 else 0
            true = int(labels[end_row])

            if true == 1:
                total_positive += 1
            if pred == 1 and true == 1:
                tp += 1
                size_err = abs(p_size.item() - size_val)
                depth_err = abs(p_depth.item() - depth_val)
                size_errors.append(size_err)
                depth_errors.append(depth_err)
                if size_err <= step_size and depth_err <= step_depth:
                    correct_within_step += 1
            elif pred == 0 and true == 0:
                tn += 1
            elif pred == 1 and true == 0:
                fp += 1
            else:
                fn += 1

            key = (size_val, depth_val)
            curr = best_samples.get(key)
            if curr is None or prob_val > curr['prob']:
                best_samples[key] = {
                    'prob': prob_val,
                    'seq': seq_norm.copy(),
                    'p_size': float(p_size.item()),
                    'p_depth': float(p_depth.item())
                }

            file_key = os.path.join(base_dir_rel, '3.CSV')
            err_score = abs(p_size.item() - size_val) + abs(p_depth.item() - depth_val)
            best_file = best_per_file.get(file_key)
            if best_file is None or err_score < best_file['err']:
                best_per_file[file_key] = {
                    'err': float(err_score),
                    'prob': prob_val,
                    'seq': seq_norm.copy(),
                    'p_size': float(p_size.item()),
                    'p_depth': float(p_depth.item()),
                    't_size': float(size_val),
                    't_depth': float(depth_val)
                }

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mae_size = float(np.mean(size_errors)) if size_errors else 0.0
    mae_depth = float(np.mean(depth_errors)) if depth_errors else 0.0

    acc_within_step = correct_within_step / total_positive if total_positive else 0.0

    per_file_records = []
    per_file_correct = 0
    for file_key, sample in best_per_file.items():
        size_err = abs(sample['p_size'] - sample['t_size'])
        depth_err = abs(sample['p_depth'] - sample['t_depth'])
        within = (size_err <= step_size) and (depth_err <= step_depth)
        if within:
            per_file_correct += 1
        per_file_records.append({
            'file': file_key,
            'true_size': sample['t_size'],
            'true_depth': sample['t_depth'],
            'pred_size': sample['p_size'],
            'pred_depth': sample['p_depth'],
            'err_size': size_err,
            'err_depth': depth_err,
            'within_one_step': within,
            'prob': sample['prob'],
            'err_sum': sample['err']
        })

    per_file_accuracy = per_file_correct / len(per_file_records) if per_file_records else 0.0

    metrics = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae_size': mae_size,
        'mae_depth': mae_depth,
        'acc_within_one_step': acc_within_step,
        'step_size': step_size,
        'step_depth': step_depth,
        'total_positive': total_positive,
        'correct_within_step': correct_within_step,
        'per_file_accuracy': per_file_accuracy,
        'per_file_total': len(per_file_records),
        'per_file_correct': per_file_correct
    }

    metrics_path = os.path.join(output_dir, 'file3_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    per_file_path = os.path.join(output_dir, 'file3_per_file_metrics.json')
    with open(per_file_path, 'w', encoding='utf-8') as f:
        json.dump(per_file_records, f, ensure_ascii=False, indent=2)

    file_keys = list(best_per_file.keys())
    file_keys.sort()
    cols = 6
    rows = int(np.ceil(len(file_keys) / cols)) if file_keys else 1
    fig = plt.figure(figsize=(22, 3.2 * rows))
    plt.suptitle('File 3 Best Match Triplets (Active DualStream)', fontsize=16, y=0.99)

    for idx, fkey in enumerate(file_keys):
        ax = plt.subplot(rows, cols, idx + 1)
        sample = best_per_file[fkey]
        seq = sample['seq'][:, 0, :, :]
        triplet = seq[-3:]
        imgs = [align_to_center(triplet[i]) for i in range(3)]
        combined = np.concatenate(imgs, axis=1)
        ax.imshow(combined, cmap='turbo', vmin=0, vmax=1)
        title = (f"S:{sample['t_size']} D:{sample['t_depth']}\n"
                 f"PS:{sample['p_size']:.2f} PD:{sample['p_depth']:.2f}\n"
                 f"E:{sample['err']:.2f}")
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    grid_path = os.path.join(output_dir, 'File3_BestMatch_Triplets.png')
    plt.savefig(grid_path, dpi=300)

    print(f"Metrics saved: {metrics_path}")
    print(f"Per-file metrics saved: {per_file_path}")
    print(f"Grid saved: {grid_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    evaluate()
