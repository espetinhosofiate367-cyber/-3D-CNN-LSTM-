import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class NoduleSequenceDataset(Dataset):
    def __init__(self, root_dir, json_label_path, mode='train', seq_len=10, transform=None, norm_method='frame', active_samples=None):
        """
        Args:
            root_dir (str): Dataset root.
            json_label_path (str): Path to manual_keyframe_labels.json.
            mode (str): 'train', 'val', 'test', 'train_mixed', 'active_train'.
            seq_len (int): Length of the sequence window.
            transform (callable, optional): Transform to apply.
            norm_method (str): 'frame' (per-frame minmax) or 'sequence' (per-sequence minmax).
            active_samples (list of tuple, optional): List of (filename, end_frame_idx) to include from File 2 in 'active_train' mode.
        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.norm_method = norm_method
        self.active_samples = set(active_samples) if active_samples else set()
        
        # We store (data_array, labels_array, full_path) for each loaded file
        self.file_data = [] 
        
        # Valid indices: list of (file_idx, end_row_idx)
        self.indices = []
        
        with open(json_label_path, 'r', encoding='utf-8') as f:
            self.labels_map = json.load(f)
            
        self._prepare_data(mode)

    def get_sample_identity(self, idx):
        """Returns (filename, end_frame_idx) for a given dataset index.
           Useful for identifying samples during active learning selection."""
        file_idx, end_row = self.indices[idx]
        _, _, full_path = self.file_data[file_idx]
        filename = os.path.basename(full_path)
        return filename, end_row
        
    def _prepare_data(self, mode):
        print(f"Preparing sequence data for mode: {mode}, seq_len={self.seq_len}...")
        
        # Helper to process a file
        def process_file(full_path, size_val, depth_val, segments=None, is_active_target=False):
            if not os.path.exists(full_path):
                return
            
            try:
                df = pd.read_csv(full_path)
                # Assume last 96 columns are data
                data_values = df.iloc[:, -96:].values.astype(np.float32)
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
                return
            
            # Data Cleaning: Check for NaN or All-Zeros
            if np.isnan(data_values).any():
                # print(f"Skipping {os.path.basename(full_path)} due to NaN values.")
                # Instead of skipping whole file, we could mask, but skipping file is safer for now if widespread
                # Or just check per sequence. Let's check per sequence later?
                # The user said "clean those missing data". 
                # If NaNs are sparse, we might handle in loop. 
                # For now, let's just replace NaNs with 0 or skip sequences with NaN.
                # Let's skip sequences with NaN in the loop below.
                pass
            
            # Create per-frame labels
            num_frames = len(data_values)
            labels = np.zeros((num_frames, 3), dtype=np.float32) # [prob, size, depth]
            
            if segments:
                is_nodule = np.zeros(num_frames, dtype=bool)
                for start, end in segments:
                    start = max(0, start)
                    end = min(num_frames, end)
                    is_nodule[start:end] = True
                    labels[start:end, 0] = 1.0
                    labels[start:end, 1] = size_val
                    labels[start:end, 2] = depth_val
            else:
                # Unlabeled file (Val/Test) or Active Learning Candidate
                # Mark prob as -1.0 to indicate unknown (unless we have labels)
                # Wait, for active_train, we are assuming we have labels (user said "1,2 have labels")
                # So if segments are passed (even for File 2), we use them.
                # If segments are None, we treat as unlabeled.
                labels[:, 0] = -1.0
                labels[:, 1] = size_val
                labels[:, 2] = depth_val
            
            # Add to storage
            file_idx = len(self.file_data)
            filename = os.path.basename(full_path)
            self.file_data.append((data_values, labels, full_path))
            
            # Generate valid indices
            # A valid sequence ends at 'i' such that [i-seq_len+1 : i+1] is valid
            
            if segments or (mode == 'active_train' and is_active_target):
                # For Training (Labeled) OR Active Learning Selection
                
                valid_indices = []
                for i in range(self.seq_len - 1, num_frames):
                    # Data Cleaning Check (Per Sequence)
                    seq_data = data_values[i - self.seq_len + 1 : i + 1]
                    if np.isnan(seq_data).any():
                        continue
                    if np.all(seq_data == 0):
                        continue
                        
                    # Active Learning Filter
                    if is_active_target:
                        # Only include if this specific frame is in active_samples
                        if (filename, i) not in self.active_samples:
                            continue
                            
                    valid_indices.append(i)
                
                # If pure training (not active target filter), do balancing
                if segments and not is_active_target and mode in ['train', 'train_mixed']:
                    pos_indices = [i for i in valid_indices if labels[i, 0] == 1.0]
                    neg_indices = [i for i in valid_indices if labels[i, 0] == 0.0]
                    
                    if len(neg_indices) > len(pos_indices) * 3:
                        if len(pos_indices) > 0:
                            stride = len(neg_indices) // (len(pos_indices) * 3)
                            if stride < 1: stride = 1
                            neg_indices = neg_indices[::stride]
                        else:
                            neg_indices = neg_indices[::10]
                    
                    for i in pos_indices:
                        self.indices.append((file_idx, i))
                    for i in neg_indices:
                        self.indices.append((file_idx, i))
                else:
                    # For active target (File 2 selected samples), just add them all
                    # (They are already filtered by active_samples)
                    for i in valid_indices:
                        self.indices.append((file_idx, i))
                    
            else:
                # For Val/Test (Unlabeled)
                stride = 1 
                for i in range(self.seq_len - 1, num_frames, stride):
                    # Data Cleaning Check
                    seq_data = data_values[i - self.seq_len + 1 : i + 1]
                    if np.isnan(seq_data).any():
                        continue
                    if np.all(seq_data == 0):
                        continue
                        
                    self.indices.append((file_idx, i))

        # Iterate over labels
        for rel_path, info in self.labels_map.items():
            size_str = info['size']
            if '2.0cm' in size_str or size_str.startswith('2cm'):
                 continue
            
            try:
                size_val = float(info['size'].replace('cm大', '').replace('cm', ''))
                depth_val = float(info['depth'].replace('cm深', '').replace('cm', ''))
            except:
                continue

            safe_rel_path = rel_path.replace('\\\\', os.sep).replace('\\', os.sep).replace('/', os.sep)
            base_dir = os.path.dirname(safe_rel_path)
            
            files_to_load = []
            if mode == 'train':
                files_to_load.append(('1.CSV', info['segments'], False))
            elif mode == 'val':
                files_to_load.append(('2.CSV', info['segments'], False)) # Load labels for File 2 (User confirmed labels exist)
            elif mode == 'test':
                files_to_load.append(('3.CSV', info['segments'], False))
            elif mode == 'train_mixed':
                files_to_load.append(('1.CSV', info['segments'], False))
                files_to_load.append(('2.CSV', info['segments'], False))
            elif mode == 'active_train':
                # File 1: Always include all (Base training set)
                files_to_load.append(('1.CSV', info['segments'], False))
                # File 2: Include only selected (is_active_target=True)
                # We assume File 2 has same segments/labels as File 1 for this experiment
                files_to_load.append(('2.CSV', info['segments'], True))
            
            for fname, segments, is_active in files_to_load:
                full_path = os.path.join(self.root_dir, base_dir, fname)
                process_file(full_path, size_val, depth_val, segments, is_active)

                     
        print(f"Mode {mode}: Loaded {len(self.indices)} sequences from {len(self.file_data)} files.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, end_row = self.indices[idx]
        data_full, labels_full, _ = self.file_data[file_idx]
        
        # Get filename info (we need to store filenames when loading)
        # But `file_data` only stores data.
        # Let's add filenames to `file_data` or a separate list.
        # Actually, self.file_data is a list of tuples.
        # Let's modify `process_file` to store filename too.
        
        # Wait, modifying `__getitem__` signature is risky if DataLoader expects standard outputs.
        # But we can return a dictionary or extra items.
        # The collate_fn handles it usually.
        # Let's check `generate_final_visualizations.py` loop:
        # for i, (data, avg_intensity, target_prob, target_size, target_depth) in enumerate(val_loader):
        # It unpacks 5 items.
        # If we add a 6th item (filename info), we need to update the loop.
        
        # Extract sequence
        # Shape: (seq_len, 96)
        seq_data = data_full[end_row - self.seq_len + 1 : end_row + 1]
        
        # Label of the last frame
        label = labels_full[end_row]
        prob, size, depth = label[0], label[1], label[2]
        
        # Preprocessing
        # Reshape to (seq_len, 1, 12, 8)
        seq_out = np.zeros((self.seq_len, 1, 12, 8), dtype=np.float32)
        
        if self.norm_method == 'sequence':
            # Sequence-level Normalization
            s_min, s_max = seq_data.min(), seq_data.max()
            if s_max - s_min > 1e-6:
                seq_data_norm = (seq_data - s_min) / (s_max - s_min)
            else:
                seq_data_norm = seq_data - s_min
                
            for i in range(self.seq_len):
                seq_out[i, 0] = seq_data_norm[i].reshape(12, 8)
        else:
            # Default: Frame-level Normalization
            for i in range(self.seq_len):
                frame = seq_data[i]
                d_min, d_max = frame.min(), frame.max()
                if d_max - d_min > 1e-6:
                    frame = (frame - d_min) / (d_max - d_min)
                else:
                    frame = frame - d_min
                seq_out[i, 0] = frame.reshape(12, 8)
            
        # Calculate Intensity Stats (Avg, Max, Std)
        avg_intensity = np.mean(seq_data)
        max_intensity = np.max(seq_data)
        std_intensity = np.std(seq_data)
        
        intensity_stats = np.array([avg_intensity, max_intensity, std_intensity], dtype=np.float32)
            
        # Return index `idx` which maps to `self.indices[idx]` -> (file_idx, end_row)
        # We can retrieve filename later using `dataset.get_file_info(idx)`
        
        return torch.tensor(seq_out), torch.tensor(intensity_stats, dtype=torch.float32), torch.tensor(prob, dtype=torch.float32), torch.tensor(size, dtype=torch.float32), torch.tensor(depth, dtype=torch.float32), idx

    def get_file_info(self, idx):
        file_idx, end_row = self.indices[idx]
        # We need to store filenames.
        # Let's modify `self.file_data` to include filename in `_prepare_data`.
        return self.file_data[file_idx][2], end_row # filename, frame_idx
