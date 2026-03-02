import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class NoduleSequenceDataset(Dataset):
    def __init__(self, root_dir, json_label_path, mode='train', seq_len=10, transform=None):
        """
        Args:
            root_dir (str): Dataset root.
            json_label_path (str): Path to manual_keyframe_labels.json.
            mode (str): 'train', 'val', 'test'.
            seq_len (int): Length of the sequence window.
            transform (callable, optional): Transform to apply.
        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        
        # We store (data_array, label_info_list) for each loaded file
        # label_info_list is a list of same length as data_array, containing dicts of {prob, size, depth}
        self.file_data = [] 
        
        # Valid indices: list of (file_idx, end_row_idx)
        self.indices = []
        
        with open(json_label_path, 'r', encoding='utf-8') as f:
            self.labels_map = json.load(f)
            
        self._prepare_data(mode)
        
    def _prepare_data(self, mode):
        print(f"Preparing sequence data for mode: {mode}, seq_len={self.seq_len}...")
        
        # Helper to process a file
        def process_file(full_path, size_val, depth_val, segments=None):
            if not os.path.exists(full_path):
                return
            
            try:
                df = pd.read_csv(full_path)
                # Assume last 96 columns are data
                data_values = df.iloc[:, -96:].values.astype(np.float32)
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
                return
            
            # Create per-frame labels
            num_frames = len(data_values)
            labels = np.zeros((num_frames, 3), dtype=np.float32) # [prob, size, depth]
            
            # Default prob is 0.0 (negative)
            # Size/Depth 0.0 for negatives
            
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
                # Unlabeled file (Val/Test)
                # Mark prob as -1.0 to indicate unknown
                labels[:, 0] = -1.0
                labels[:, 1] = size_val
                labels[:, 2] = depth_val
            
            # Add to storage
            file_idx = len(self.file_data)
            self.file_data.append((data_values, labels))
            
            # Generate valid indices
            # A valid sequence ends at 'i' such that [i-seq_len+1 : i+1] is valid
            # So i must be >= seq_len - 1
            
            if segments:
                # For Training (Labeled):
                # We want to balance Positives and Negatives.
                # Positives: frames where label[i, 0] == 1.0
                # Negatives: frames where label[i, 0] == 0.0
                
                valid_indices = []
                for i in range(self.seq_len - 1, num_frames):
                    # The label of the sequence is the label of the LAST frame
                    # This is standard for "real-time detection" simulation
                    valid_indices.append(i)
                
                pos_indices = [i for i in valid_indices if labels[i, 0] == 1.0]
                neg_indices = [i for i in valid_indices if labels[i, 0] == 0.0]
                
                # Downsample negatives?
                # Let's keep a ratio, say 1:1 or 1:2.
                # Or just keep all if dataset is small. 
                # Given the previous training was fast (seconds), dataset is small.
                # But let's check counts. 
                # If too many negatives, simple stride.
                if len(neg_indices) > len(pos_indices) * 3:
                    # Stride to reduce negatives
                    stride = len(neg_indices) // (len(pos_indices) * 3)
                    if stride < 1: stride = 1
                    neg_indices = neg_indices[::stride]
                
                # Add to self.indices
                for i in pos_indices:
                    self.indices.append((file_idx, i))
                for i in neg_indices:
                    self.indices.append((file_idx, i))
                    
            else:
                # For Val/Test (Unlabeled):
                # Add all valid windows (maybe with stride to save time if needed)
                stride = 1 # Evaluate every frame
                for i in range(self.seq_len - 1, num_frames, stride):
                    self.indices.append((file_idx, i))

        # Iterate over labels
        for rel_path, info in self.labels_map.items():
            size_str = info['size']
            # Filter 2.0cm (Exclude)
            # Include 0.25cm
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
                files_to_load.append(('1.CSV', info['segments']))
            elif mode == 'val':
                files_to_load.append(('2.CSV', None))
            elif mode == 'test':
                files_to_load.append(('3.CSV', None))
            
            for fname, segments in files_to_load:
                full_path = os.path.join(self.root_dir, base_dir, fname)
                process_file(full_path, size_val, depth_val, segments)
                     
        print(f"Mode {mode}: Loaded {len(self.indices)} sequences from {len(self.file_data)} files.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, end_row = self.indices[idx]
        data_full, labels_full = self.file_data[file_idx]
        
        # Extract sequence
        # Shape: (seq_len, 96)
        seq_data = data_full[end_row - self.seq_len + 1 : end_row + 1]
        
        # Label of the last frame
        label = labels_full[end_row]
        prob, size, depth = label[0], label[1], label[2]
        
        # Preprocessing
        # Reshape to (seq_len, 1, 12, 8)
        # Normalize: Min-Max per frame? Or per sequence?
        # Usually per-frame is robust for varying pressure.
        # But per-sequence preserves temporal relative changes.
        # Let's do per-frame normalization to be safe against base pressure drift.
        
        seq_out = np.zeros((self.seq_len, 1, 12, 8), dtype=np.float32)
        
        for i in range(self.seq_len):
            frame = seq_data[i]
            d_min, d_max = frame.min(), frame.max()
            if d_max - d_min > 1e-6:
                frame = (frame - d_min) / (d_max - d_min)
            else:
                frame = frame - d_min
            seq_out[i, 0] = frame.reshape(12, 8)
            
        # Calculate Average Intensity (Raw)
        # We can take the mean of the whole sequence or just the last frame.
        # Since depth is a property of the nodule (which is present in the sequence),
        # and pressure varies, maybe mean of the sequence is more stable.
        # However, the user said "Combine overall average stress intensity".
        # Let's use the mean of the entire sequence data (raw).
        # seq_data is the raw data slice.
        avg_intensity = np.mean(seq_data)
            
        return torch.tensor(seq_out), torch.tensor(avg_intensity, dtype=torch.float32), torch.tensor(prob, dtype=torch.float32), torch.tensor(size, dtype=torch.float32), torch.tensor(depth, dtype=torch.float32)
