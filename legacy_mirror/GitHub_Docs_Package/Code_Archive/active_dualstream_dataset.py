import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActiveDualStreamDataset(Dataset):
    def __init__(self, root_dir, base_labels_path, manual_labels_path=None, mode='train_base', seq_len=10):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.file_data = []
        self.indices = []

        with open(base_labels_path, 'r', encoding='utf-8') as f:
            self.base_labels = json.load(f)

        self.manual_labels = None
        if manual_labels_path and os.path.exists(manual_labels_path):
            with open(manual_labels_path, 'r', encoding='utf-8') as f:
                self.manual_labels = json.load(f)

        self._prepare_data(mode)

    def _load_csv(self, full_path):
        df = pd.read_csv(full_path)
        data_values = df.iloc[:, -96:].values.astype(np.float32)
        return data_values

    def _build_labels(self, num_frames, size_val, depth_val, segments=None, labeled=True):
        labels = np.zeros((num_frames, 3), dtype=np.float32)
        if segments:
            is_nodule = np.zeros(num_frames, dtype=bool)
            for start, end in segments:
                start = max(0, int(start))
                end = min(num_frames, int(end))
                is_nodule[start:end] = True
                labels[start:end, 0] = 1.0
                labels[start:end, 1] = size_val
                labels[start:end, 2] = depth_val
        else:
            labels[:, 0] = 0.0 if labeled else -1.0
            labels[:, 1] = size_val
            labels[:, 2] = depth_val
        return labels

    def _add_indices(self, file_idx, labels, num_frames, labeled=True):
        valid_indices = list(range(self.seq_len - 1, num_frames))
        if labeled:
            pos_indices = [i for i in valid_indices if labels[i, 0] == 1.0]
            neg_indices = [i for i in valid_indices if labels[i, 0] == 0.0]
            if len(pos_indices) == 0:
                target = min(1000, len(neg_indices)) if len(neg_indices) > 0 else 0
                stride = max(1, len(neg_indices) // target) if target > 0 else 1
                neg_indices = neg_indices[::stride]
            elif len(neg_indices) > len(pos_indices) * 3:
                stride = len(neg_indices) // (len(pos_indices) * 3)
                if stride < 1:
                    stride = 1
                neg_indices = neg_indices[::stride]
            for i in pos_indices:
                self.indices.append((file_idx, i))
            for i in neg_indices:
                self.indices.append((file_idx, i))
        else:
            for i in valid_indices:
                self.indices.append((file_idx, i))

    def _process_entry(self, rel_path, info, file_name, segments, labeled=True):
        try:
            size_val = float(info['size'].replace('cm大', '').replace('cm', ''))
            depth_val = float(info['depth'].replace('cm深', '').replace('cm', ''))
        except Exception:
            return

        safe_rel = rel_path.replace('\\', os.sep).replace('/', os.sep)
        base_dir = os.path.dirname(safe_rel)
        full_path = os.path.join(self.root_dir, base_dir, file_name)
        if not os.path.exists(full_path):
            return
        data_values = self._load_csv(full_path)
        num_frames = len(data_values)
        labels = self._build_labels(num_frames, size_val, depth_val, segments=segments, labeled=labeled)
        file_idx = len(self.file_data)
        self.file_data.append((data_values, labels))
        self._add_indices(file_idx, labels, num_frames, labeled=labeled)

    def _prepare_data(self, mode):
        if mode == 'train_base':
            labels_map = self.base_labels
            for rel_path, info in labels_map.items():
                segments = info.get('segments', None)
                self._process_entry(rel_path, info, '1.CSV', segments, labeled=True)
        elif mode == 'val':
            labels_map = self.manual_labels if self.manual_labels else self.base_labels
            for rel_path, info in labels_map.items():
                segments = info.get('segments', None) if self.manual_labels else None
                self._process_entry(rel_path, info, '2.CSV', segments, labeled=True)
        elif mode == 'train_active':
            for rel_path, info in self.base_labels.items():
                segments = info.get('segments', None)
                self._process_entry(rel_path, info, '1.CSV', segments, labeled=True)
            if self.manual_labels:
                for rel_path, info in self.manual_labels.items():
                    segments = info.get('segments', None)
                    self._process_entry(rel_path, info, '2.CSV', segments, labeled=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, end_row = self.indices[idx]
        data_full, labels_full = self.file_data[file_idx]
        seq_data = data_full[end_row - self.seq_len + 1: end_row + 1]
        label = labels_full[end_row]
        prob, size, depth = label[0], label[1], label[2]

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

        return torch.tensor(seq_out), torch.tensor(stats_seq), torch.tensor(prob, dtype=torch.float32), torch.tensor(size, dtype=torch.float32), torch.tensor(depth, dtype=torch.float32)


class UnlabeledSequenceDataset(Dataset):
    def __init__(self, root_dir, file_name='2.CSV', seq_len=10):
        self.root_dir = root_dir
        self.file_name = file_name
        self.seq_len = seq_len
        self.file_data = []
        self.indices = []

        labels_path = os.path.join(root_dir, 'manual_keyframe_labels_file2.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels_map = json.load(f)
        else:
            labels_map = {}

        for rel_path, info in labels_map.items():
            safe_rel = rel_path.replace('\\', os.sep).replace('/', os.sep)
            base_dir = os.path.dirname(safe_rel)
            full_path = os.path.join(self.root_dir, base_dir, file_name)
            if not os.path.exists(full_path):
                continue
            df = pd.read_csv(full_path)
            data_values = df.iloc[:, -96:].values.astype(np.float32)
            file_idx = len(self.file_data)
            self.file_data.append((data_values, full_path))
            for i in range(self.seq_len - 1, len(data_values)):
                self.indices.append((file_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, end_row = self.indices[idx]
        data_full, path = self.file_data[file_idx]
        seq_data = data_full[end_row - self.seq_len + 1: end_row + 1]
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

        return torch.tensor(seq_out), torch.tensor(stats_seq), path, end_row
