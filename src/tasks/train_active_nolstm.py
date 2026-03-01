import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from refactored_project.src.core.datasets import ActiveDualStreamDataset
from refactored_project.src.core.model_factory import build_model


def train_or_eval(model, loader, device, criterion_cls, criterion_reg, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for seq, stats_seq, prob, size, depth in loader:
            seq = seq.to(device)
            stats_seq = stats_seq.to(device)
            prob = prob.to(device).unsqueeze(1)
            size = size.to(device).unsqueeze(1)
            depth = depth.to(device).unsqueeze(1)

            if is_train:
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
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * seq.size(0)
    return total_loss / max(1, len(loader.dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-base", type=int, default=10)
    parser.add_argument("--epochs-active", type=int, default=15)
    args = parser.parse_args()

    root_dir = os.environ.get("TACTILE_DATA_ROOT", "")
    base_labels = os.environ.get("BASE_LABELS_JSON", "")
    manual_labels = os.environ.get("FILE2_LABELS_JSON", "")
    output_dir = os.environ.get("OUTPUT_DIR", "")
    if not all([root_dir, base_labels, manual_labels, output_dir]):
        raise RuntimeError("请先设置 TACTILE_DATA_ROOT/BASE_LABELS_JSON/FILE2_LABELS_JSON/OUTPUT_DIR")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("nolstm", seq_len=10).to(device)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_base = ActiveDualStreamDataset(root_dir, base_labels, mode="train_base", seq_len=10)
    val_base = ActiveDualStreamDataset(root_dir, base_labels, manual_labels_path=manual_labels, mode="val", seq_len=10)
    train_loader = DataLoader(train_base, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_base, batch_size=32, shuffle=False)

    history = []
    best_val = float("inf")
    best_base_path = os.path.join(output_dir, "dualstream_3dcnn_nolstm_base_refactor.pth")
    for epoch in range(args.epochs_base):
        train_loss = train_or_eval(model, train_loader, device, criterion_cls, criterion_reg, optimizer=optimizer)
        val_loss = train_or_eval(model, val_loader, device, criterion_cls, criterion_reg)
        history.append({"stage": "base", "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_base_path)
        print(f"[NoLSTM-BASE] {epoch+1}/{args.epochs_base} train={train_loss:.4f} val={val_loss:.4f}")

    model.load_state_dict(torch.load(best_base_path, map_location=device))
    train_active = ActiveDualStreamDataset(root_dir, base_labels, manual_labels_path=manual_labels, mode="train_active", seq_len=10)
    train_loader = DataLoader(train_active, batch_size=32, shuffle=True)

    best_val = float("inf")
    best_active_path = os.path.join(output_dir, "dualstream_3dcnn_nolstm_active_refactor.pth")
    for epoch in range(args.epochs_active):
        train_loss = train_or_eval(model, train_loader, device, criterion_cls, criterion_reg, optimizer=optimizer)
        val_loss = train_or_eval(model, val_loader, device, criterion_cls, criterion_reg)
        history.append({"stage": "active", "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_active_path)
        print(f"[NoLSTM-ACTIVE] {epoch+1}/{args.epochs_active} train={train_loss:.4f} val={val_loss:.4f}")

    with open(os.path.join(output_dir, "train_active_nolstm_refactor_log.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
