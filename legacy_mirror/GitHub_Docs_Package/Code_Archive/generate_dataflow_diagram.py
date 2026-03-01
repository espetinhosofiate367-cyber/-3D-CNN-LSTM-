import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def box(ax, xy, text, w=2.6, h=0.8):
    rect = FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02', linewidth=1.2, edgecolor='#2F3B52', facecolor='#E8F1FF')
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center', fontsize=10)


def arrow(ax, start, end):
    arr = FancyArrowPatch(start, end, arrowstyle=ArrowStyle('->', head_length=6, head_width=3), linewidth=1.2, color='#2F3B52')
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box(ax, (0.6, 8.4), '传感器/CSV')
    box(ax, (0.6, 6.6), '序列构造\n10帧')
    box(ax, (0.6, 4.8), '归一化\n统计特征')
    box(ax, (4.0, 6.6), '3D CNN\n形态特征')
    box(ax, (4.0, 4.8), 'LSTM\n时序聚合')
    box(ax, (7.4, 5.7), '融合/输出\n概率+尺寸+深度')

    arrow(ax, (2.0, 8.4), (2.0, 7.4))
    arrow(ax, (2.0, 6.6), (2.0, 5.6))
    arrow(ax, (3.2, 5.2), (4.0, 6.0))
    arrow(ax, (3.2, 5.2), (4.0, 5.2))
    arrow(ax, (6.6, 6.6), (7.4, 6.1))
    arrow(ax, (6.6, 4.8), (7.4, 5.3))

    ax.set_title('输入输出数据流程（CNN3D->LSTM 方案）', fontsize=12)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'active_learning_results', 'dataflow_cnn3d_lstm.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(out_path)


if __name__ == '__main__':
    main()
