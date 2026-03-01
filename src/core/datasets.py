import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActiveDualStreamDataset(Dataset):
    def __init__(self, root_dir, base_labels_path, manual_labels_path=None, mode="train_base", seq_len=10):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.file_data = []
        self.indices = []

        with open(base_labels_path, "r", encoding="utf-8") as f:
            self.base_labels = json.load(f)

        self.manual_labels = None
        if manual_labels_path and os.path.exists(manual_labels_path):
            with open(manual_labels_path, "r", encoding="utf-8") as f:
                self.manual_labels = json.load(f)

        self._prepare_data(mode)

    @staticmethod
    def _load_csv(full_path):
        df = pd.read_csv(full_path)
        return df.iloc[:, -96:].values.astype(np.float32)

    @staticmethod
    def _build_labels(num_frames, size_val, depth_val, segments=None):
        labels = np.zeros((num_frames, 3), dtype=np.float32)
        if segments:
            for start, end in segments:
                start = max(0, int(start))
                end = min(num_frames, int(end))
                labels[start:end, 0] = 1.0
                labels[start:end, 1] = size_val
                labels[start:end, 2] = depth_val
        return labels

    def _add_indices(self, file_idx, labels, num_frames):
        valid_indices = list(range(self.seq_len - 1, num_frames))
        pos_indices = [i for i in valid_indices if labels[i, 0] == 1.0]
        neg_indices = [i for i in valid_indices if labels[i, 0] == 0.0]
        if len(pos_indices) > 0 and len(neg_indices) > len(pos_indices) * 3:
            stride = max(1, len(neg_indices) // (len(pos_indices) * 3))
            neg_indices = neg_indices[::stride]
        self.indices.extend((file_idx, i) for i in pos_indices)
        self.indices.extend((file_idx, i) for i in neg_indices)

    def _process_entry(self, rel_path, info, file_name, segments):
        try:
            size_val = float(info["size"].replace("cm大", "").replace("cm", ""))
            depth_val = float(info["depth"].replace("cm深", "").replace("cm", ""))
        except Exception:
            return

        safe_rel = rel_path.replace("\\", os.sep).replace("/", os.sep)
        base_dir = os.path.dirname(safe_rel)
        full_path = os.path.join(self.root_dir, base_dir, file_name)
        if not os.path.exists(full_path):
            return

        data_values = self._load_csv(full_path)
        labels = self._build_labels(len(data_values), size_val, depth_val, segments=segments)
        file_idx = len(self.file_data)
        self.file_data.append((data_values, labels))
        self._add_indices(file_idx, labels, len(data_values))

    def _prepare_data(self, mode):
        if mode == "train_base":
            for rel_path, info in self.base_labels.items():
                self._process_entry(rel_path, info, "1.CSV", info.get("segments", None))
        elif mode == "train_active":
            for rel_path, info in self.base_labels.items():
                self._process_entry(rel_path, info, "1.CSV", info.get("segments", None))
            if self.manual_labels:
                for rel_path, info in self.manual_labels.items():
                    self._process_entry(rel_path, info, "2.CSV", info.get("segments", None))
        elif mode == "val":
            labels_map = self.manual_labels if self.manual_labels else self.base_labels
            for rel_path, info in labels_map.items():
                self._process_entry(rel_path, info, "2.CSV", info.get("segments", None))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, end_row = self.indices[idx]
        data_full, labels_full = self.file_data[file_idx]
        seq_data = data_full[end_row - self.seq_len + 1 : end_row + 1]
        prob, size, depth = labels_full[end_row]

        seq_out = np.zeros((self.seq_len, 1, 12, 8), dtype=np.float32)
        stats_seq = np.zeros((self.seq_len, 3), dtype=np.float32)
        for i in range(self.seq_len):
            frame = seq_data[i]
            d_min, d_max = frame.min(), frame.max()
            if d_max - d_min > 1e-6:
                norm = (frame - d_min) / (d_max - d_min)
            else:
                norm = frame - d_min
            seq_out[i, 0] = norm.reshape(12, 8)
            stats_seq[i, 0] = float(np.mean(frame))
            stats_seq[i, 1] = float(np.max(frame))
            stats_seq[i, 2] = float(np.std(frame))

        return (
            torch.tensor(seq_out),
            torch.tensor(stats_seq),
            torch.tensor(prob, dtype=torch.float32),
            torch.tensor(size, dtype=torch.float32),
            torch.tensor(depth, dtype=torch.float32),
        )
