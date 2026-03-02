import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from refactored_project.src.core.model_factory import build_model, model_weight_name


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_data(path):
    df = pd.read_csv(path)
    return df.iloc[:, -96:].values.astype(np.float32)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dualstream", "nolstm"], default="dualstream")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--step-depth", type=float, default=0.5)
    args = parser.parse_args()

    data_root = os.environ.get("TACTILE_DATA_ROOT", "")
    labels_path = os.environ.get("BASE_LABELS_JSON", "")
    output_dir = os.environ.get("OUTPUT_DIR", "")
    if not all([data_root, labels_path, output_dir]):
        raise RuntimeError("请先设置 TACTILE_DATA_ROOT/BASE_LABELS_JSON/OUTPUT_DIR")

    model_path = os.path.join(output_dir, model_weight_name(args.model))
    model = build_model(args.model, seq_len=10)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    labels_map = load_labels(labels_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tp = tn = fp = fn = 0
    size_errors = []
    depth_errors = []
    per_file_records = []

    for rel_path, info in labels_map.items():
        try:
            size_val = float(info["size"].replace("cm大", "").replace("cm", ""))
            depth_val = float(info["depth"].replace("cm深", "").replace("cm", ""))
        except Exception:
            continue

        safe_rel = rel_path.replace("\\", os.sep).replace("/", os.sep)
        base_dir_rel = os.path.dirname(safe_rel)
        file3_path = os.path.join(data_root, base_dir_rel, "3.CSV")
        if not os.path.exists(file3_path):
            continue

        data = read_csv_data(file3_path)
        labels = build_label_array(len(data), info.get("segments", []))

        best_hit = None
        best_hit_score = -1.0
        positive_seen = False

        for end_row in range(9, len(data)):
            seq_raw = data[end_row - 9 : end_row + 1]
            if np.isnan(seq_raw).any() or np.all(seq_raw == 0):
                continue

            seq_norm = normalize_sequence(seq_raw)
            stats = stats_sequence(seq_raw)

            input_tensor = torch.tensor(seq_norm).unsqueeze(0).to(device)
            stats_tensor = torch.tensor(stats).unsqueeze(0).to(device)
            with torch.no_grad():
                prob, p_size, p_depth = model(input_tensor, stats_tensor)

            prob_val = float(torch.sigmoid(prob).item())
            pred = 1 if prob_val >= args.threshold else 0
            true = int(labels[end_row])
            if true == 1:
                positive_seen = True

            if pred == 1 and true == 1:
                tp += 1
                size_errors.append(abs(p_size.item() - size_val))
                depth_errors.append(abs(p_depth.item() - depth_val))
            elif pred == 0 and true == 0:
                tn += 1
            elif pred == 1 and true == 0:
                fp += 1
            else:
                fn += 1

            if prob_val > best_hit_score:
                best_hit_score = prob_val
                best_hit = {
                    "file": str(os.path.join(base_dir_rel, "3.CSV")).replace("\\", "/"),
                    "true_size": size_val,
                    "true_depth": depth_val,
                    "pred_size": float(p_size.item()),
                    "pred_depth": float(p_depth.item()),
                    "prob": prob_val,
                }

        if positive_seen and best_hit is not None:
            size_err = abs(best_hit["pred_size"] - best_hit["true_size"])
            depth_err = abs(best_hit["pred_depth"] - best_hit["true_depth"])
            within_one_step = (size_err <= args.step_size) and (depth_err <= args.step_depth)
            best_hit["err_size"] = size_err
            best_hit["err_depth"] = depth_err
            best_hit["within_one_step"] = bool(within_one_step)
            per_file_records.append(best_hit)

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "model": args.model,
        "threshold": args.threshold,
        "step_size": args.step_size,
        "step_depth": args.step_depth,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae_size": float(np.mean(size_errors)) if size_errors else 0.0,
        "mae_depth": float(np.mean(depth_errors)) if depth_errors else 0.0,
    }

    file_total = len(per_file_records)
    file_correct = int(sum(1 for r in per_file_records if r["within_one_step"]))
    metrics["file_level_total"] = file_total
    metrics["file_level_correct"] = file_correct
    metrics["file_level_detection_rate"] = (file_correct / file_total) if file_total else 0.0

    out_path = os.path.join(output_dir, f"file3_metrics_refactor_{args.model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    out_file_level = os.path.join(output_dir, f"file3_per_file_metrics_refactor_{args.model}.json")
    with open(out_file_level, "w", encoding="utf-8") as f:
        json.dump(per_file_records, f, ensure_ascii=False, indent=2)

    print(out_path)
    print(out_file_level)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
