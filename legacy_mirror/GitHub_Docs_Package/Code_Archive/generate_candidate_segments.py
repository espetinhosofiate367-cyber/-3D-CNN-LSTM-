import os
import json
import numpy as np
import torch
from dualstream_3dcnn_lstm import DualStream3DCNNLSTM


def load_candidates(path):
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


def compute_probs(model, data, device, seq_len=10):
    n = len(data)
    probs = np.full((n,), np.nan, dtype=np.float32)
    if n < seq_len:
        return probs
    model.eval()
    with torch.no_grad():
        for end_row in range(seq_len - 1, n):
            window = data[end_row - seq_len + 1: end_row + 1]
            seq_out = np.zeros((seq_len, 1, 12, 8), dtype=np.float32)
            stats = np.zeros((seq_len, 3), dtype=np.float32)
            for i in range(seq_len):
                frame = window[i]
                d_min, d_max = frame.min(), frame.max()
                if d_max - d_min > 1e-6:
                    frame_norm = (frame - d_min) / (d_max - d_min)
                else:
                    frame_norm = frame - d_min
                seq_out[i, 0] = frame_norm.reshape(12, 8)
                stats[i] = np.array([np.mean(frame), np.max(frame), np.std(frame)], dtype=np.float32)

            input_tensor = torch.tensor(seq_out).unsqueeze(0).to(device)
            stats_tensor = torch.tensor(stats).unsqueeze(0).to(device)
            prob, _, _ = model(input_tensor, stats_tensor)
            probs[end_row] = torch.sigmoid(prob).item()
    return probs


def expand_segment(probs, center, min_window=5, max_window=30):
    n = len(probs)
    center = int(center)
    if center < 0 or center >= n:
        return None, None

    local_start = max(0, center - max_window)
    local_end = min(n - 1, center + max_window)
    local_probs = probs[local_start:local_end + 1]
    valid = local_probs[np.isfinite(local_probs)]
    if len(valid) == 0:
        return max(0, center - min_window), min(n - 1, center + min_window)

    local_max = float(np.max(valid))
    threshold = max(0.3, 0.6 * local_max)

    start = center
    end = center
    while start - 1 >= 0 and np.isfinite(probs[start - 1]) and probs[start - 1] >= threshold:
        start -= 1
    while end + 1 < n and np.isfinite(probs[end + 1]) and probs[end + 1] >= threshold:
        end += 1

    if end - start < min_window:
        start = max(0, center - min_window)
        end = min(n - 1, center + min_window)
    return start, end


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates_path = os.path.join(base_dir, 'active_learning_results', 'active_learning_candidates.json')
    model_path = os.path.join(base_dir, 'active_learning_results', 'dualstream_3dcnn_lstm_base.pth')
    output_path = os.path.join(base_dir, 'active_learning_results', 'active_learning_candidates_segments.json')

    if not os.path.exists(candidates_path):
        print(f"候选文件不存在: {candidates_path}")
        return
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStream3DCNNLSTM(seq_len=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    candidates = load_candidates(candidates_path)
    by_file = {}
    for item in candidates:
        path = os.path.normpath(item.get('file_path', ''))
        if not path:
            continue
        by_file.setdefault(path, []).append(item)

    for path, items in by_file.items():
        if not os.path.exists(path):
            continue
        data = read_csv_data(path)
        probs = compute_probs(model, data, device, seq_len=10)
        for item in items:
            center = int(item.get('end_row', 0))
            start, end = expand_segment(probs, center)
            item['segment_start'] = int(start) if start is not None else None
            item['segment_end'] = int(end) if end is not None else None

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print(f"候选片段已保存: {output_path}")


if __name__ == '__main__':
    main()
