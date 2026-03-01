import json
import os

import matplotlib.pyplot as plt


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    output_dir = os.environ.get("OUTPUT_DIR", "")
    if not output_dir or not os.path.exists(output_dir):
        raise RuntimeError("请先设置有效的 OUTPUT_DIR")

    dual = load_json(os.path.join(output_dir, "file3_metrics_refactor_dualstream.json"))
    nolstm = load_json(os.path.join(output_dir, "file3_metrics_refactor_nolstm.json"))
    if dual is None or nolstm is None:
        raise FileNotFoundError("未找到双模型评估结果，请先运行 evaluate_file3.py")

    models = ["DualStream", "NoLSTM"]
    mae_size = [dual["mae_size"], nolstm["mae_size"]]
    mae_depth = [dual["mae_depth"], nolstm["mae_depth"]]
    det_rate = [dual.get("file_level_detection_rate", 0.0), nolstm.get("file_level_detection_rate", 0.0)]

    plt.figure(figsize=(8, 5))
    x = [0, 1]
    plt.bar([i - 0.15 for i in x], mae_size, width=0.3, label="MAE Size")
    plt.bar([i + 0.15 for i in x], mae_depth, width=0.3, label="MAE Depth")
    plt.xticks(x, models)
    plt.ylabel("Error (cm)")
    plt.title("MAE Comparison (File3)")
    plt.legend()
    mae_path = os.path.join(output_dir, "comparison_mae_size_depth.png")
    plt.tight_layout()
    plt.savefig(mae_path, dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(models, det_rate)
    plt.ylim(0, 1)
    plt.ylabel("Detection Rate")
    plt.title("File-level Detection Rate Comparison")
    det_path = os.path.join(output_dir, "comparison_file_level_detection_rate.png")
    plt.tight_layout()
    plt.savefig(det_path, dpi=180)
    plt.close()

    print(mae_path)
    print(det_path)


if __name__ == "__main__":
    main()
