import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sequence_dataset import NoduleSequenceDataset
from final_model import DualStreamModel
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_ROOT = r"c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\数据集 最新\建表数据"
LABEL_PATH = r"c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\manual_keyframe_labels.json"
MODEL_PATH = r"c:\Users\SWH\Desktop\智能医疗检测系统\Princess_Pea_Release\models\best_model.pth"
SEQ_LEN = 10
OUTPUT_DIR = r"c:\Users\SWH\Desktop\智能医疗检测系统\Princess_Pea_Release\results"

def calculate_detection_rate():
    print("="*50)
    print("Princess & Pea: Detection Rate Calculation & Visualization (File 3)")
    print("="*50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Dataset
    print("Loading Test Dataset (File 3)...")
    dataset = NoduleSequenceDataset(
        root_dir=DATA_ROOT,
        json_label_path=LABEL_PATH,
        mode='test',
        seq_len=SEQ_LEN,
        norm_method='frame'
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Total Samples: {len(dataset)}")

    # 2. Load Model
    print("Loading Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamModel(seq_len=SEQ_LEN).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 3. Collect Data
    # Group by filename: {filename: [frame_results]}
    file_predictions = defaultdict(list)
    
    # Store all ground truth values to find unique gradients
    true_sizes_all = []
    true_depths_all = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in loader:
            seq_data, intensity_stats, prob_label, size_label, depth_label, idx = batch
            
            # Get filename
            dataset_idx = idx.item()
            filename, frame_idx = dataset.get_file_info(dataset_idx)
            
            # Inference
            seq_data = seq_data.float().to(device)
            intensity_stats = intensity_stats.float().to(device)
            
            prob, size, depth = model(seq_data, intensity_stats)
            
            result = {
                'frame_idx': frame_idx,
                'true_prob': prob_label.item(),
                'true_size': size_label.item(),
                'true_depth': depth_label.item(),
                'pred_prob': prob.item(),
                'pred_size': size.item(),
                'pred_depth': depth.item()
            }
            
            file_predictions[filename].append(result)
            
            if prob_label.item() > 0.5:
                true_sizes_all.append(size_label.item())
                true_depths_all.append(depth_label.item())

    # 4. Analyze Gradients
    unique_sizes = sorted(list(set(true_sizes_all)))
    unique_depths = sorted(list(set(true_depths_all)))
    
    print("\n--- Gradient Analysis ---")
    if len(unique_sizes) > 1:
        size_diffs = np.diff(unique_sizes)
        size_gradient = np.min(size_diffs[size_diffs > 0]) if len(size_diffs) > 0 else 0.25
    else:
        size_gradient = 0.25 

    if len(unique_depths) > 1:
        depth_diffs = np.diff(unique_depths)
        depth_gradient = np.min(depth_diffs[depth_diffs > 0]) if len(depth_diffs) > 0 else 0.5
    else:
        depth_gradient = 0.5 

    print(f"Calculated Size Gradient (Threshold): {size_gradient:.2f} cm")
    print(f"Calculated Depth Gradient (Threshold): {depth_gradient:.2f} cm")

    # 5. Calculate File-Level Detection Rate
    
    file_stats = [] # Store {'filename', 'true_size', 'true_depth', 'detected'}

    print("\n--- File-Level Evaluation ---")
    print(f"Criteria: At least one frame in file has:")
    print(f"  1. Pred_Prob > 0.5")
    print(f"  2. |Pred_Size - True_Size| < {size_gradient:.2f} cm")
    print(f"  3. |Pred_Depth - True_Depth| < {depth_gradient:.2f} cm")

    for filename, frames in file_predictions.items():
        # Check if positive file
        is_positive_file = any(f['true_prob'] > 0.5 for f in frames)
        if not is_positive_file:
            continue
            
        # Get Ground Truth for this file (assuming constant for positive frames)
        # Use the first positive frame to get GT
        positive_frames = [f for f in frames if f['true_prob'] > 0.5]
        if not positive_frames:
            continue # Should not happen given is_positive_file check
            
        true_size = positive_frames[0]['true_size']
        true_depth = positive_frames[0]['true_depth']
        
        # Check Detection
        file_detected = False
        for f in frames:
            is_high_prob = f['pred_prob'] > 0.5
            is_size_acc = abs(f['pred_size'] - f['true_size']) < size_gradient
            is_depth_acc = abs(f['pred_depth'] - f['true_depth']) < depth_gradient
            
            if is_high_prob and is_size_acc and is_depth_acc:
                file_detected = True
                break
        
        file_stats.append({
            'filename': filename,
            'Size (cm)': round(true_size, 2), # Round to avoid float precision issues in grouping
            'Depth (cm)': round(true_depth, 2),
            'Detected': 1 if file_detected else 0
        })

    # 6. Report & Visualization
    df = pd.DataFrame(file_stats)
    
    total_files = len(df)
    detected_files = df['Detected'].sum()
    overall_rate = (detected_files / total_files) * 100
    
    print("\n" + "="*50)
    print("FINAL RESULTS (FILE LEVEL)")
    print("="*50)
    print(f"Total Positive Files: {total_files}")
    print(f"Successfully Detected: {detected_files}")
    print(f"Detection Rate: {overall_rate:.2f}%")
    
    # --- Visualization ---
    try:
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Princess & Pea Detection Performance (File-Level)\nTotal Files: {total_files} | Overall Detection Rate: {overall_rate:.1f}%", fontsize=16)

        # 1. Overall Pie Chart
        plt.subplot(2, 2, 1)
        plt.pie([detected_files, total_files - detected_files], 
                labels=['Detected', 'Missed'], 
                autopct='%1.1f%%', 
                colors=['#66b3ff', '#ff9999'], 
                explode=(0.05, 0), 
                startangle=90)
        plt.title("Overall Detection Rate")

        # 2. Detection Rate by Size
        plt.subplot(2, 2, 2)
        size_group = df.groupby('Size (cm)')['Detected'].mean() * 100
        size_counts = df.groupby('Size (cm)')['Detected'].count()
        
        ax2 = sns.barplot(x=size_group.index, y=size_group.values, palette="Blues_d")
        plt.ylabel("Detection Rate (%)")
        plt.title("Detection Rate by Nodule Size")
        plt.ylim(0, 105)
        # Add sample counts on top
        for i, p in enumerate(ax2.patches):
            ax2.annotate(f'n={size_counts.iloc[i]}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # 3. Detection Rate by Depth
        plt.subplot(2, 2, 3)
        depth_group = df.groupby('Depth (cm)')['Detected'].mean() * 100
        depth_counts = df.groupby('Depth (cm)')['Detected'].count()
        
        ax3 = sns.barplot(x=depth_group.index, y=depth_group.values, palette="Greens_d")
        plt.ylabel("Detection Rate (%)")
        plt.title("Detection Rate by Nodule Depth")
        plt.ylim(0, 105)
        for i, p in enumerate(ax3.patches):
            ax3.annotate(f'n={depth_counts.iloc[i]}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # 4. Heatmap (Size vs Depth) - Detection Count/Total
        # We want to see which specific combinations failed
        plt.subplot(2, 2, 4)
        pivot_table = df.pivot_table(index='Size (cm)', columns='Depth (cm)', values='Detected', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1)
        plt.title("Detection Rate Heatmap (Size vs Depth)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(OUTPUT_DIR, "detection_rate_analysis.png")
        plt.savefig(save_path, dpi=300)
        print(f"\nVisualization saved to: {save_path}")
        
    except Exception as e:
        print(f"Error generating charts: {e}")

if __name__ == "__main__":
    calculate_detection_rate()
