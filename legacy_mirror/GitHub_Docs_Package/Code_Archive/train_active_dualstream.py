import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dualstream_3dcnn_lstm import DualStream3DCNNLSTM
from active_dualstream_dataset import ActiveDualStreamDataset, UnlabeledSequenceDataset


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def train_one_epoch(model, loader, device, optimizer, criterion_cls, criterion_reg):
    model.train()
    total_loss = 0.0
    for seq, stats_seq, prob, size, depth in loader:
        seq = seq.to(device)
        stats_seq = stats_seq.to(device)
        prob = prob.to(device).unsqueeze(1)
        size = size.to(device).unsqueeze(1)
        depth = depth.to(device).unsqueeze(1)

        optimizer.zero_grad()
        prob_pred, size_pred, depth_pred = model(seq, stats_seq)

        loss_prob = criterion_cls(prob_pred, (prob > 0.5).float())
        mask = torch.sigmoid(prob_pred) > 0.5
        if mask.sum() > 0:
            loss_size = criterion_reg(size_pred[mask], size[mask])
            loss_depth = criterion_reg(depth_pred[mask], depth[mask])
        else:
            loss_size = torch.tensor(0.0, device=device)
            loss_depth = torch.tensor(0.0, device=device)

        loss = loss_prob + loss_size + loss_depth
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq.size(0)
    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, device, criterion_cls, criterion_reg):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for seq, stats_seq, prob, size, depth in loader:
            seq = seq.to(device)
            stats_seq = stats_seq.to(device)
            prob = prob.to(device).unsqueeze(1)
            size = size.to(device).unsqueeze(1)
            depth = depth.to(device).unsqueeze(1)

            prob_pred, size_pred, depth_pred = model(seq, stats_seq)

            loss_prob = criterion_cls(prob_pred, (prob > 0.5).float())
            mask = torch.sigmoid(prob_pred) > 0.5
            if mask.sum() > 0:
                loss_size = criterion_reg(size_pred[mask], size[mask])
                loss_depth = criterion_reg(depth_pred[mask], depth[mask])
            else:
                loss_size = torch.tensor(0.0, device=device)
                loss_depth = torch.tensor(0.0, device=device)
            loss = loss_prob + loss_size + loss_depth
            total_loss += loss.item() * seq.size(0)
    return total_loss / len(loader.dataset)


def select_uncertain_samples(model, loader, device, top_k=200):
    model.eval()
    scores = []
    with torch.no_grad():
        for seq, stats_seq, path, end_row in loader:
            seq = seq.to(device)
            stats_seq = stats_seq.to(device)
            prob_pred, _, _ = model(seq, stats_seq)
            prob = prob_pred.squeeze(1).cpu().numpy()
            for i in range(len(prob)):
                score = abs(prob[i] - 0.5)
                scores.append({
                    'file_path': path[i],
                    'end_row': int(end_row[i]),
                    'prob': float(prob[i]),
                    'uncertainty': float(score)
                })
    scores.sort(key=lambda x: x['uncertainty'])
    return scores[:top_k]


def main():
    root_dir = os.environ.get('TACTILE_DATA_ROOT', '')
    if not root_dir:
        root_dir = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\数据集 最新\建表数据'

    base_labels = os.environ.get('BASE_LABELS_JSON', '')
    if not base_labels:
        base_labels = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\manual_keyframe_labels.json'

    manual_labels = os.environ.get('FILE2_LABELS_JSON', '')
    if not manual_labels:
        manual_labels = os.path.join(root_dir, 'manual_keyframe_labels_file2.json')

    output_dir = os.environ.get('OUTPUT_DIR', '')
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'active_learning_results')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStream3DCNNLSTM(seq_len=10).to(device)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_base = ActiveDualStreamDataset(root_dir, base_labels, manual_labels_path=None, mode='train_base', seq_len=10)
    val_base = ActiveDualStreamDataset(root_dir, base_labels, manual_labels_path=None, mode='val', seq_len=10)

    train_loader = DataLoader(train_base, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_base, batch_size=32, shuffle=False)

    best_val = float('inf')
    best_path = os.path.join(output_dir, 'dualstream_3dcnn_lstm_base.pth')

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion_cls, criterion_reg)
        val_loss = eval_one_epoch(model, val_loader, device, criterion_cls, criterion_reg)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        print(f"Epoch {epoch+1}/10 | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if not os.path.exists(manual_labels):
        unlabeled = UnlabeledSequenceDataset(root_dir, file_name='2.CSV', seq_len=10)
        unl_loader = DataLoader(unlabeled, batch_size=32, shuffle=False)
        model.load_state_dict(torch.load(best_path, map_location=device))
        candidates = select_uncertain_samples(model, unl_loader, device, top_k=200)
        save_json(os.path.join(output_dir, 'active_learning_candidates.json'), candidates)
        print("请先手动标注File2，再重新运行训练。")
        return

    train_active = ActiveDualStreamDataset(root_dir, base_labels, manual_labels_path=manual_labels, mode='train_active', seq_len=10)
    train_loader = DataLoader(train_active, batch_size=32, shuffle=True)

    best_val = float('inf')
    best_path = os.path.join(output_dir, 'dualstream_3dcnn_lstm_active.pth')
    for epoch in range(15):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion_cls, criterion_reg)
        val_loss = eval_one_epoch(model, val_loader, device, criterion_cls, criterion_reg)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        print(f"Active Epoch {epoch+1}/15 | Train {train_loss:.4f} | Val {val_loss:.4f}")

    print(f"模型已保存: {best_path}")


if __name__ == '__main__':
    main()
