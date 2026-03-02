import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, text, w=2.6, h=0.85, color="#E8F1FF"):
    rect = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", linewidth=1.2, edgecolor="#2F3B52", facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, start, end):
    arr = FancyArrowPatch(start, end, arrowstyle="->", linewidth=1.2, color="#2F3B52", mutation_scale=12)
    ax.add_patch(arr)


def main():
    output_dir = os.environ.get("OUTPUT_DIR", "")
    if not output_dir:
        raise RuntimeError("请先设置 OUTPUT_DIR")
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(ax, (0.5, 8.5), "CSV读取\n(96维/帧)")
    box(ax, (0.5, 6.8), "滑窗构造\nT=10")
    box(ax, (0.5, 5.1), "归一化\n12x8")
    box(ax, (0.5, 3.4), "统计特征\nmean/max/std")

    box(ax, (4.2, 6.8), "形态流\n3DConv+BN+Pool")
    box(ax, (4.2, 5.1), "时序折叠\n(T->C)")
    box(ax, (4.2, 3.4), "SpatialConv+GAP")

    box(ax, (8.0, 5.1), "统计流\nFC+BiLSTM")
    box(ax, (10.8, 5.1), "融合层\nConcat+FC")
    box(ax, (10.8, 3.4), "多任务头\nProb/Size/Depth", color="#FFF0E8")

    arrow(ax, (1.8, 8.5), (1.8, 7.65))
    arrow(ax, (1.8, 6.8), (1.8, 5.95))
    arrow(ax, (1.8, 5.1), (1.8, 4.25))
    arrow(ax, (3.1, 5.45), (4.2, 7.15))
    arrow(ax, (3.1, 3.8), (8.0, 5.5))
    arrow(ax, (5.5, 6.8), (5.5, 5.95))
    arrow(ax, (5.5, 5.1), (5.5, 4.25))
    arrow(ax, (6.8, 3.8), (10.8, 5.5))
    arrow(ax, (9.3, 5.5), (10.8, 5.5))
    arrow(ax, (12.1, 5.1), (12.1, 4.25))

    ax.set_title("内部数据流升级图（重构版）", fontsize=12)
    out_path = os.path.join(output_dir, "internal_dataflow_upgraded.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
