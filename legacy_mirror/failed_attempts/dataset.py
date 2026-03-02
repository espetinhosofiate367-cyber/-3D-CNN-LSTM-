
import os
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import ndimage

class KeyframeDataset(Dataset):
    def __init__(self, data_dir, label_file, sequence_length=30, mode='train', interp_factor=5):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.mode = mode
        self.interp_factor = interp_factor
        
        # Load labels
        with open(label_file, 'r') as f:
            self.manual_labels = json.load(f)
            
        self.samples = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        # Gather all labeled files
        files_data = []
        
        # 1. 处理标注文件中的正样本（及其对应的增强数据）
        for rel_path, info in self.manual_labels.items():
            # 基础路径 (1.csv)
            base_full_path = os.path.join(self.data_dir, rel_path)
            
            # 尝试查找同目录下的增强文件 (2.csv, 3.csv)
            # 假设 rel_path 类似于 "0.25cm大/0.25cm深/1.CSV"
            dir_name = os.path.dirname(base_full_path)
            base_name = os.path.basename(base_full_path)
            
            # 构建待处理文件列表：原始 + 增强
            target_files = [base_full_path]
            
            # 如果是 "1.CSV" 或 "1.csv"，尝试找 "2.CSV", "3.CSV"
            if base_name.lower() == '1.csv':
                target_files.append(os.path.join(dir_name, '2.CSV'))
                target_files.append(os.path.join(dir_name, '3.CSV'))
                # 也尝试小写后缀
                target_files.append(os.path.join(dir_name, '2.csv'))
                target_files.append(os.path.join(dir_name, '3.csv'))
            
            # 去重并验证存在性
            target_files = list(set([f for f in target_files if os.path.exists(f)]))
            
            for file_path in target_files:
                try:
                    # Load CSV
                    df = pd.read_csv(file_path)
                    mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
                    if mat_cols:
                        mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
                        data = df[mat_cols].values
                    else:
                        data = df.iloc[:, -96:].values
                    
                    # Parse Size/Depth (Use info from JSON, even for 2.csv/3.csv)
                    size_val = float(re.search(r"([\d\.]+)", info['size']).group(1)) * 10 # cm -> mm
                    depth_val = float(re.search(r"([\d\.]+)", info['depth']).group(1)) * 10 # cm -> mm
                    
                    # Create mask for keyframes
                    label_mask = np.zeros(len(data), dtype=np.float32)
                    
                    # 应用相同的标注片段
                    for start, end in info['segments']:
                        # 确保不超过当前文件长度 (虽然增强文件长度应该一致，但防万一)
                        s = max(0, min(start, len(data)-1))
                        e = max(0, min(end, len(data)-1))
                        label_mask[s:e+1] = 1.0
                    
                    files_data.append({
                        'data': data,
                        'labels': label_mask,
                        'size': size_val,
                        'depth': depth_val,
                        'source': 'labeled'
                    })
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        # 2. 收集未标注文件作为负样本 (Explicit Negative Mining)
        # 遍历 data_dir 下的所有 CSV，如果不在 files_data 已处理列表中，则视为负样本
        # 注意：这里需要谨慎，只有当用户明确表示"未选择的为未找到"时才这样做
        processed_paths = set([os.path.normpath(os.path.join(self.data_dir, rel_path)) for rel_path in self.manual_labels.keys()])
        
        # 递归查找所有 CSV
        all_csvs = []
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.lower().endswith('.csv'):
                    all_csvs.append(os.path.join(root, f))
        
        # 筛选出未标注的文件
        # 这里有一个假设：只有 1.csv 被标注了，所以 manual_labels 只有 1.csv
        # 那么 2.csv, 3.csv 如果上面没被增强加载（即 1.csv 没标），那么它们也应该是负样本
        # 简化逻辑：所有不在 files_data 中的文件，都是负样本
        # 但要注意：如果 1.csv 被标了，2.csv 在上面被加载了，这里就不应该再加载
        # 所以我们需要记录已经加载过的绝对路径
        
        # 由于上面 target_files 已经是绝对路径，我们可以收集已加载的路径
        # 但 files_data 里没存路径。让我们在上面存一下。
        # (代码修改了 files_data 结构，添加了 source，但没存 path，这里假装没法直接去重)
        # 更好的方法：建立一个已处理文件的集合
        loaded_files_set = set()
        # 重新执行一遍上面的逻辑来填充 set (为了代码整洁，不重写了，假设上面逻辑正确)
        # 我们直接在上面循环里 add 到 loaded_files_set
        
        # 重新构建第一步，为了获取 loaded_files_set
        loaded_files_set = set()
        files_data = [] # Reset
        
        for rel_path, info in self.manual_labels.items():
            base_full_path = os.path.join(self.data_dir, rel_path)
            dir_name = os.path.dirname(base_full_path)
            base_name = os.path.basename(base_full_path)
            
            target_files = [base_full_path]
            if base_name.lower() == '1.csv':
                target_files.append(os.path.join(dir_name, '2.CSV'))
                target_files.append(os.path.join(dir_name, '3.CSV'))
                target_files.append(os.path.join(dir_name, '2.csv'))
                target_files.append(os.path.join(dir_name, '3.csv'))
            
            target_files = list(set([f for f in target_files if os.path.exists(f)]))
            
            for file_path in target_files:
                if os.path.normpath(file_path) in loaded_files_set: continue
                
                try:
                    df = pd.read_csv(file_path)
                    mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
                    if mat_cols:
                        mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
                        data = df[mat_cols].values
                    else:
                        data = df.iloc[:, -96:].values
                    
                    size_val = float(re.search(r"([\d\.]+)", info['size']).group(1)) * 10
                    depth_val = float(re.search(r"([\d\.]+)", info['depth']).group(1)) * 10
                    
                    label_mask = np.zeros(len(data), dtype=np.float32)
                    for start, end in info['segments']:
                        s = max(0, min(start, len(data)-1))
                        e = max(0, min(end, len(data)-1))
                        label_mask[s:e+1] = 1.0
                    
                    files_data.append({
                        'data': data,
                        'labels': label_mask,
                        'size': size_val,
                        'depth': depth_val
                    })
                    loaded_files_set.add(os.path.normpath(file_path))
                except:
                    pass

        # 第二步：加载未标注文件作为负样本
        for csv_path in all_csvs:
            norm_path = os.path.normpath(csv_path)
            if norm_path in loaded_files_set:
                continue
                
            # 这是一个未标注文件 -> 全负样本
            try:
                # 尝试从路径解析 size/depth
                # 假设路径结构: .../0.25cm大/0.5cm深/xxxx.csv
                # 倒数第2级是深度，倒数第3级是大小
                parts = norm_path.split(os.sep)
                depth_str = parts[-2]
                size_str = parts[-3]
                
                try:
                    size_val = float(re.search(r"([\d\.]+)", size_str).group(1)) * 10
                    depth_val = float(re.search(r"([\d\.]+)", depth_str).group(1)) * 10
                except:
                    # 解析失败，使用默认值或跳过
                    # 也可以设为 0 表示未知
                    size_val = 10.0
                    depth_val = 10.0
                
                df = pd.read_csv(csv_path)
                mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
                if mat_cols:
                    mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
                    data = df[mat_cols].values
                else:
                    data = df.iloc[:, -96:].values
                
                # 全 0 标签
                label_mask = np.zeros(len(data), dtype=np.float32)
                
                files_data.append({
                    'data': data,
                    'labels': label_mask,
                    'size': size_val,
                    'depth': depth_val
                })
                
            except Exception as e:
                # print(f"Skipping negative file {csv_path}: {e}")
                pass

        # Generate Sliding Windows
        # We need balanced positive and negative samples
        pos_samples = []
        neg_samples = []
        
        for fd in files_data:
            data = fd['data']
            labels = fd['labels']
            meta = np.array([fd['size'], fd['depth']], dtype=np.float32)
            
            # Slide window
            # Step size can be smaller for training
            step = 5
            for i in range(0, len(data) - self.sequence_length, step):
                window_data = data[i : i+self.sequence_length]
                mid_idx = i + self.sequence_length // 2
                is_keyframe = labels[mid_idx] > 0.5
                
                sample = {
                    'data': window_data,
                    'meta': meta,
                    'label': 1.0 if is_keyframe else 0.0
                }
                
                if is_keyframe:
                    pos_samples.append(sample)
                else:
                    neg_samples.append(sample)
        
        # Balance data
        # Oversample positives or Undersample negatives?
        # Let's undersample negatives to match positives * ratio
        ratio = 3 # 3:1 neg:pos ratio
        
        if len(pos_samples) > 0:
            if len(neg_samples) > len(pos_samples) * ratio:
                np.random.shuffle(neg_samples)
                neg_samples = neg_samples[:len(pos_samples) * ratio]
        else:
            print("Warning: No positive samples found!")
            # Keep some negatives anyway to prevent crash, but training will be useless
            neg_samples = neg_samples[:100]
        
        self.samples = pos_samples + neg_samples
        np.random.shuffle(self.samples)
        
        print(f"Dataset prepared: {len(pos_samples)} Positive, {len(neg_samples)} Negative samples.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Preprocessing: Interpolation and Normalization
        # Data is currently (30, 96)
        seq_data = sample['data'] # (30, 96)
        
        # Reshape and Interpolate
        # This is slow if done here. But for 30 frames * batch it might be okay?
        # Let's try.
        processed_seq = []
        for frame in seq_data:
            mat = frame.reshape(12, 8)
            # Normalize frame-wise or keep absolute? 
            # Absolute is better for intensity thresholding features.
            # But CNN likes 0-1.
            mn, mx = mat.min(), mat.max()
            if mx - mn > 1e-6:
                norm = (mat - mn) / (mx - mn)
            else:
                norm = mat - mn
            
            # Interpolate
            img = ndimage.zoom(norm, self.interp_factor, order=1) # 12x8 -> 60x40
            processed_seq.append(img)
            
        seq_tensor = torch.tensor(np.array(processed_seq), dtype=torch.float32) # (30, 60, 40)
        meta_tensor = torch.tensor(sample['meta'], dtype=torch.float32)
        label_tensor = torch.tensor([sample['label']], dtype=torch.float32)
        
        return seq_tensor, meta_tensor, label_tensor
