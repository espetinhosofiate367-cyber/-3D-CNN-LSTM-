import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dualstream = r'c:\Users\SWH\Desktop\智能医疗检测系统\核心程序\deep_learning\active_learning_results\file3_eval\file3_metrics.json'
    nolstm = os.path.join(base_dir, 'active_learning_results', 'file3_eval_nolstm', 'file3_metrics.json')
    cnn3d_lstm = os.path.join(base_dir, 'active_learning_results', 'file3_eval_cnn3d_lstm', 'file3_metrics.json')

    m_dual = load_metrics(dualstream)
    m_nolstm = load_metrics(nolstm)
    m_cnn = load_metrics(cnn3d_lstm)

    entries = [
        ('DualStream', m_dual),
        ('NoLSTM', m_nolstm),
        ('CNN3D->LSTM', m_cnn)
    ]

    labels = []
    size_mae = []
    depth_mae = []
    for name, metrics in entries:
        if not metrics:
            continue
        size_val = metrics['size_mae'] if 'size_mae' in metrics else metrics.get('mae_size')
        depth_val = metrics['depth_mae'] if 'depth_mae' in metrics else metrics.get('mae_depth')
        if size_val is None or depth_val is None:
            continue
        labels.append(name)
        size_mae.append(size_val)
        depth_mae.append(depth_val)

    x = range(len(labels))
    fig = plt.figure(figsize=(7, 4))
    plt.bar([i - 0.15 for i in x], size_mae, width=0.3, label='Size MAE')
    plt.bar([i + 0.15 for i in x], depth_mae, width=0.3, label='Depth MAE')
    plt.xticks(list(x), labels)
    plt.ylabel('MAE (cm)')
    plt.title('Ablation Comparison (File3)')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(base_dir, 'active_learning_results', 'ablation_compare_mae.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(out_path)


if __name__ == '__main__':
    main()
