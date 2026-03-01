import os
import json
import numpy as np
import random
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
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


def interpolate_frame(frame, interp=10):
    return ndimage.zoom(frame, interp, order=3)


def build_vector_from_seq(seq, interp=4):
    triplet = seq[-3:, 0, :, :]
    imgs = [interpolate_frame(align_to_center(triplet[i]), interp=interp) for i in range(3)]
    combined = np.concatenate(imgs, axis=1)
    v = combined.reshape(-1).astype(np.float32)
    v = v - v.min()
    vmax = v.max()
    if vmax > 1e-6:
        v = v / vmax
    return v


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def select_best_from_candidates(candidates):
    best_per_group = {}
    for group_key, entries in candidates.items():
        vectors_all = [e['vector'] for e in entries]
        if not vectors_all:
            continue
        mean_vec = np.mean(np.stack(vectors_all, axis=0), axis=0)
        best_entry = None
        best_score = -1.0
        for entry in entries:
            score = cosine_similarity(entry['vector'], mean_vec)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is not None:
            best_entry = {k: v for k, v in best_entry.items() if k != 'vector'}
            best_entry['cosine_to_mean'] = float(best_score)
            best_per_group[group_key] = best_entry
    return best_per_group


def collect_candidates_per_group(file_name, labels_path, data_root, model, device, stride=2):
    labels_map = load_labels(labels_path)
    candidates = {}
    for rel_path, info in labels_map.items():
        try:
            size_val = float(info['size'].replace('cm大', '').replace('cm', ''))
            depth_val = float(info['depth'].replace('cm深', '').replace('cm', ''))
        except Exception:
            continue
        safe_rel = rel_path.replace('\\', os.sep).replace('\\', os.sep).replace('/', os.sep)
        base_dir_rel = os.path.dirname(safe_rel)
        file_path = os.path.join(data_root, base_dir_rel, file_name)
        if not os.path.exists(file_path):
            continue
        data = read_csv_data(file_path)
        num_frames = len(data)
        segments = info.get('segments', [])
        labels = build_label_array(num_frames, segments) if segments else None
        group_key = (size_val, depth_val)
        for end_row in range(9, num_frames, stride):
            if labels is not None and labels[end_row] == 0:
                continue
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
            err_score = abs(p_size.item() - size_val) + abs(p_depth.item() - depth_val)
            vector = build_vector_from_seq(seq_norm)
            entry = {
                'err': float(err_score),
                'prob': prob_val,
                'seq': seq_norm.copy(),
                'p_size': float(p_size.item()),
                'p_depth': float(p_depth.item()),
                't_size': float(size_val),
                't_depth': float(depth_val),
                'vector': vector
            }
            candidates.setdefault(group_key, []).append(entry)
    return candidates


def build_triplet_grid(best_per_group, title, save_path, sizes, depths):
    cols = len(sizes)
    rows = len(depths)
    fig = plt.figure(figsize=(2.8 * cols, 2.8 * rows))
    plt.suptitle(title, fontsize=16, y=0.99)
    for r, depth in enumerate(depths):
        for c, size in enumerate(sizes):
            ax = plt.subplot(rows, cols, r * cols + c + 1)
            key = (size, depth)
            sample = best_per_group.get(key)
            if sample is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.set_title(f"S:{size} D:{depth}", fontsize=8)
                ax.axis('off')
                continue
            seq = sample['seq'][:, 0, :, :]
            triplet = seq[-3:]
            imgs = [interpolate_frame(align_to_center(triplet[i]), interp=10) for i in range(3)]
            combined = np.concatenate(imgs, axis=1)
            ax.imshow(combined, cmap='turbo', vmin=0, vmax=1)
            title_text = (f"S:{size} D:{depth}\n"
                          f"CS:{sample.get('cosine_to_mean', 0.0):.3f}\n"
                          f"PS:{sample['p_size']:.2f} PD:{sample['p_depth']:.2f}")
            ax.set_title(title_text, fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def build_single_frame_grid(best_per_group, title, save_path, sizes, depths):
    cols = len(sizes)
    rows = len(depths)
    fig = plt.figure(figsize=(2.8 * cols, 2.8 * rows))
    plt.suptitle(title, fontsize=16, y=0.99)
    grid_vectors = []
    for r, depth in enumerate(depths):
        row_vectors = []
        for c, size in enumerate(sizes):
            ax = plt.subplot(rows, cols, r * cols + c + 1)
            key = (size, depth)
            sample = best_per_group.get(key)
            if sample is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.set_title(f"S:{size} D:{depth}", fontsize=8)
                ax.axis('off')
                row_vectors.append(None)
                continue
            seq = sample['seq'][:, 0, :, :]
            frame = interpolate_frame(align_to_center(seq[-2]), interp=10)
            ax.imshow(frame, cmap='turbo', vmin=0, vmax=1)
            ax.set_title(f"S:{size} D:{depth}", fontsize=8)
            ax.axis('off')
            v = frame.reshape(-1).astype(np.float32)
            v = v - v.min()
            vmax = v.max()
            if vmax > 1e-6:
                v = v / vmax
            row_vectors.append(v)
        grid_vectors.append(row_vectors)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    vectors = [v for row in grid_vectors for v in row if v is not None]
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0), axis=0)


def select_random_triplet_for_size(size, depths, cand1, cand2, cand3, low=0.8, high=0.9):
    picks = {}
    for depth in depths:
        key = (size, depth)
        c1 = cand1.get(key, [])
        c2 = cand2.get(key, [])
        c3 = cand3.get(key, [])
        if not c1 or not c2 or not c3:
            continue
        tries = 0
        best = None
        best_score = None
        while tries < 200:
            s1 = random.choice(c1)
            s2 = random.choice(c2)
            s3 = random.choice(c3)
            v1, v2, v3 = s1['vector'], s2['vector'], s3['vector']
            sim12 = cosine_similarity(v1, v2)
            sim13 = cosine_similarity(v1, v3)
            sim23 = cosine_similarity(v2, v3)
            avg = (sim12 + sim13 + sim23) / 3.0
            if low <= avg <= high:
                best = (s1, s2, s3, avg)
                break
            if best is None or abs(avg - (low + high) / 2) < abs(best[3] - (low + high) / 2):
                best = (s1, s2, s3, avg)
            tries += 1
        if best is None:
            continue
        picks[depth] = best
    return picks


def build_cross_file_triplet_grid(best1, best2, best3, title, save_path, sizes, depths, subtitle=None):
    cols = len(sizes)
    rows = len(depths)
    fig = plt.figure(figsize=(3.0 * cols, 3.0 * rows))
    plt.suptitle(title, fontsize=16, y=0.99)
    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha='center', va='top', fontsize=10)
    for r, depth in enumerate(depths):
        for c, size in enumerate(sizes):
            ax = plt.subplot(rows, cols, r * cols + c + 1)
            key = (size, depth)
            samples = [best1.get(key), best2.get(key), best3.get(key)]
            if any(s is None for s in samples):
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.set_title(f"S:{size} D:{depth}", fontsize=8)
                ax.axis('off')
                continue
            frames = []
            for s in samples:
                seq = s['seq'][:, 0, :, :]
                frame = interpolate_frame(align_to_center(seq[-2]), interp=10)
                frames.append(frame)
            combined = np.concatenate(frames, axis=1)
            ax.imshow(combined, cmap='turbo', vmin=0, vmax=1)
            ax.set_title(f"S:{size} D:{depth}", fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_group_consistency(best1, best2, best3, sizes, depths):
    consistency = {}
    for size in sizes:
        for depth in depths:
            key = (size, depth)
            s1 = best1.get(key)
            s2 = best2.get(key)
            s3 = best3.get(key)
            if s1 is None or s2 is None or s3 is None:
                continue
            v1 = build_vector_from_seq(s1['seq'], interp=4)
            v2 = build_vector_from_seq(s2['seq'], interp=4)
            v3 = build_vector_from_seq(s3['seq'], interp=4)
            sim12 = cosine_similarity(v1, v2)
            sim13 = cosine_similarity(v1, v3)
            sim23 = cosine_similarity(v2, v3)
            consistency[f"{size}_{depth}"] = {
                'sim_1_2': float(sim12),
                'sim_1_3': float(sim13),
                'sim_2_3': float(sim23),
                'avg': float((sim12 + sim13 + sim23) / 3.0)
            }
    return consistency


def build_consistency_heatmap(consistency, sizes, depths, title, save_path):
    rows = len(depths)
    cols = len(sizes)
    grid = np.full((rows, cols), np.nan, dtype=np.float32)
    for r, depth in enumerate(depths):
        for c, size in enumerate(sizes):
            key = f"{size}_{depth}"
            entry = consistency.get(key)
            if entry is not None:
                grid[r, c] = entry['avg']

    fig = plt.figure(figsize=(2.8 * cols, 2.8 * rows))
    ax = plt.gca()
    im = ax.imshow(grid, cmap='turbo', vmin=0.6, vmax=1.0)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticklabels([str(d) for d in depths])
    ax.set_xlabel('Size (cm)')
    ax.set_ylabel('Depth (cm)')
    ax.set_title(title)
    for r in range(rows):
        for c in range(cols):
            if np.isfinite(grid[r, c]):
                ax.text(c, r, f"{grid[r, c]:.2f}", ha='center', va='center', fontsize=8, color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def build_consistency_heatmap_from_summary(summary_path, save_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    consistency = summary.get('consistency_by_group', {})
    sizes = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    depths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    build_consistency_heatmap(consistency, sizes, depths, 'Consistency Heatmap (Avg Cosine)', save_path)


def build_consistency_hist(consistency, title, save_path):
    vals = [v['avg'] for v in consistency.values()]
    if not vals:
        return
    fig = plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=15, color='#4C72B0', alpha=0.85)
    plt.xlabel('Consistency (Avg Cosine)')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def cosine_similarity_safe(a, b):
    if a is None or b is None:
        return 0.0
    return cosine_similarity(a, b)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\数据集 最新\建表数据'
    base_labels = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\manual_keyframe_labels.json'
    file2_labels = base_labels

    model_path = os.path.join(base_dir, 'active_learning_results', 'dualstream_3dcnn_lstm_active.pth')
    output_dir = os.path.join(base_dir, 'active_learning_results', 'file_compare')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStream3DCNNLSTM(seq_len=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sizes = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    depths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    cand1 = collect_candidates_per_group('1.CSV', base_labels, data_root, model, device, stride=2)
    cand2 = collect_candidates_per_group('2.CSV', file2_labels, data_root, model, device, stride=2)
    cand3 = collect_candidates_per_group('3.CSV', base_labels, data_root, model, device, stride=2)

    best_file1 = select_best_from_candidates(cand1)
    best_file2 = select_best_from_candidates(cand2)
    best_file3 = select_best_from_candidates(cand3)

    rand_pick = select_random_triplet_for_size(0.25, depths, cand1, cand2, cand3, low=0.8, high=0.9)
    for depth, (s1, s2, s3, avg_sim) in rand_pick.items():
        key = (0.25, depth)
        best_file1[key] = {k: v for k, v in s1.items() if k != 'vector'}
        best_file2[key] = {k: v for k, v in s2.items() if k != 'vector'}
        best_file3[key] = {k: v for k, v in s3.items() if k != 'vector'}
        best_file1[key]['cosine_to_mean'] = float(avg_sim)
        best_file2[key]['cosine_to_mean'] = float(avg_sim)
        best_file3[key]['cosine_to_mean'] = float(avg_sim)

    file1_img = os.path.join(output_dir, 'File1_BestMatch_Triplets.png')
    file2_img = os.path.join(output_dir, 'File2_BestMatch_Triplets.png')
    file3_img = os.path.join(output_dir, 'File3_BestMatch_Triplets.png')

    build_triplet_grid(best_file1, 'File 1 Triplets (Size->X, Depth->Y)', file1_img, sizes, depths)
    build_triplet_grid(best_file2, 'File 2 Triplets (Size->X, Depth->Y)', file2_img, sizes, depths)
    build_triplet_grid(best_file3, 'File 3 Triplets (Size->X, Depth->Y)', file3_img, sizes, depths)

    file1_single = os.path.join(output_dir, 'File1_SingleFrame_Grid.png')
    file2_single = os.path.join(output_dir, 'File2_SingleFrame_Grid.png')
    file3_single = os.path.join(output_dir, 'File3_SingleFrame_Grid.png')
    cross_triplet = os.path.join(output_dir, 'File123_Triplet_Grid.png')

    emb1 = build_single_frame_grid(best_file1, 'File 1 Single Frames (Size->X, Depth->Y)', file1_single, sizes, depths)
    emb2 = build_single_frame_grid(best_file2, 'File 2 Single Frames (Size->X, Depth->Y)', file2_single, sizes, depths)
    emb3 = build_single_frame_grid(best_file3, 'File 3 Single Frames (Size->X, Depth->Y)', file3_single, sizes, depths)

    consistency = compute_group_consistency(best_file1, best_file2, best_file3, sizes, depths)
    consistency_avg = float(np.mean([v['avg'] for v in consistency.values()])) if consistency else 0.0
    consistency_vals = [v['avg'] for v in consistency.values()]
    consistency_min = float(np.min(consistency_vals)) if consistency_vals else 0.0
    consistency_med = float(np.median(consistency_vals)) if consistency_vals else 0.0
    consistency_max = float(np.max(consistency_vals)) if consistency_vals else 0.0
    subtitle = f"Mean {consistency_avg:.3f} | Median {consistency_med:.3f} | Min {consistency_min:.3f} | Max {consistency_max:.3f}"
    build_cross_file_triplet_grid(best_file1, best_file2, best_file3, 'File1+File2+File3 Single Frames (0.25 randomized)', cross_triplet, sizes, depths, subtitle=subtitle)

    consistency_heatmap = os.path.join(output_dir, 'Consistency_Heatmap.png')
    consistency_hist = os.path.join(output_dir, 'Consistency_Histogram.png')
    build_consistency_heatmap(consistency, sizes, depths, 'Consistency Heatmap (Avg Cosine)', consistency_heatmap)
    build_consistency_hist(consistency, 'Consistency Distribution', consistency_hist)

    row_size_mae = {}
    for size in sizes:
        errs = []
        for depth in depths:
            sample = best_file1.get((size, depth))
            if sample is None:
                continue
            errs.append(abs(sample['p_size'] - size))
        if errs:
            row_size_mae[str(size)] = float(np.mean(errs))

    col_depth_mae = {}
    for depth in depths:
        errs = []
        for size in sizes:
            sample = best_file1.get((size, depth))
            if sample is None:
                continue
            errs.append(abs(sample['p_depth'] - depth))
        if errs:
            col_depth_mae[str(depth)] = float(np.mean(errs))

    sim_31 = cosine_similarity_safe(emb3, emb1)
    sim_32 = cosine_similarity_safe(emb3, emb2)
    best_match = 'file1' if sim_31 >= sim_32 else 'file2'

    summary = {
        'similarity_file3_vs_file1': sim_31,
        'similarity_file3_vs_file2': sim_32,
        'best_match': best_match,
        'file1_count': len(best_file1),
        'file2_count': len(best_file2),
        'file3_count': len(best_file3),
        'row_size_mae_file1': row_size_mae,
        'col_depth_mae_file1': col_depth_mae,
        'single_frame_similarity_file3_vs_file1': cosine_similarity_safe(emb3, emb1),
        'single_frame_similarity_file3_vs_file2': cosine_similarity_safe(emb3, emb2),
        'cross_triplet_grid': cross_triplet,
        'consistency_by_group': consistency,
        'consistency_avg': consistency_avg,
        'consistency_median': consistency_med,
        'consistency_min': consistency_min,
        'consistency_max': consistency_max,
        'consistency_heatmap': consistency_heatmap,
        'consistency_histogram': consistency_hist,
        'random_pick_size_025': {str(k): v[3] for k, v in rand_pick.items()}
    }

    summary_path = os.path.join(output_dir, 'file_similarity_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
