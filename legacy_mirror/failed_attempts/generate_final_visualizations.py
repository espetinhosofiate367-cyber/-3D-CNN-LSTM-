import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import torch
import sys

# Add path for deep learning module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Up 2 levels: paper_results -> 实验数据 -> root
dl_path = os.path.join(project_root, '核心程序', 'deep_learning')
sys.path.append(dl_path)

from sequence_dataset import NoduleSequenceDataset
from final_model import DualStreamModel
from torch.utils.data import DataLoader

def generate_paper_visualizations():
    # 1. Setup Directories
    BASE_OUTPUT_DIR = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\paper_results\final_visualizations'
    if os.path.exists(BASE_OUTPUT_DIR):
        shutil.rmtree(BASE_OUTPUT_DIR)
    os.makedirs(BASE_OUTPUT_DIR)
    
    STATS_DIR = os.path.join(BASE_OUTPUT_DIR, 'statistics')
    SAMPLES_DIR = os.path.join(BASE_OUTPUT_DIR, 'best_samples')
    os.makedirs(STATS_DIR)
    os.makedirs(SAMPLES_DIR)
    
    # 2. Load Prediction Data
    pred_path = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\paper_results\final_model_predictions.csv'
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} not found. Please run train_final.py first.")
        return
        
    df = pd.read_csv(pred_path)
    
    # 3. Generate Statistical Charts
    print("Generating statistical charts...")
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 3.1 Scatter Plots (Size & Depth)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='True Size', y='Pred Size', data=df, alpha=0.6, color='blue')
    min_val = min(df['True Size'].min(), df['Pred Size'].min())
    max_val = max(df['True Size'].max(), df['Pred Size'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.title('Size Prediction')
    plt.xlabel('True Size (cm)')
    plt.ylabel('Predicted Size (cm)')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='True Depth', y='Pred Depth', data=df, alpha=0.6, color='green')
    min_val = min(df['True Depth'].min(), df['Pred Depth'].min())
    max_val = max(df['True Depth'].max(), df['Pred Depth'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.title('Depth Prediction')
    plt.xlabel('True Depth (cm)')
    plt.ylabel('Predicted Depth (cm)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, 'scatter_plots.png'), dpi=300)
    
    # 3.2 Bland-Altman Plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    mean_size = (df['Pred Size'] + df['True Size']) / 2
    diff_size = df['Pred Size'] - df['True Size']
    md = np.mean(diff_size)
    sd = np.std(diff_size)
    plt.scatter(mean_size, diff_size, alpha=0.5, color='blue')
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='r', linestyle=':')
    plt.axhline(md - 1.96*sd, color='r', linestyle=':')
    plt.title(f'Bland-Altman: Size (Mean Diff={md:.3f})')
    plt.xlabel('Mean Size (cm)')
    plt.ylabel('Diff (Pred - True)')
    
    plt.subplot(1, 2, 2)
    mean_depth = (df['Pred Depth'] + df['True Depth']) / 2
    diff_depth = df['Pred Depth'] - df['True Depth']
    md = np.mean(diff_depth)
    sd = np.std(diff_depth)
    plt.scatter(mean_depth, diff_depth, alpha=0.5, color='green')
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='r', linestyle=':')
    plt.axhline(md - 1.96*sd, color='r', linestyle=':')
    plt.title(f'Bland-Altman: Depth (Mean Diff={md:.3f})')
    plt.xlabel('Mean Depth (cm)')
    plt.ylabel('Diff (Pred - True)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, 'bland_altman.png'), dpi=300)
    
    # 4. Find Best Samples and Visualize
    print("Finding best samples and generating detailed visualizations...")
    
    # We need to access the actual data to plot the heatmaps
    # Re-load dataset (Validation split of File 1, consistent with train_final.py)
    ROOT_DIR = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\数据集 最新\建表数据'
    JSON_PATH = r'c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\manual_keyframe_labels.json'
    SEQ_LEN = 10
    
    full_dataset = NoduleSequenceDataset(ROOT_DIR, JSON_PATH, mode='train', seq_len=SEQ_LEN)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Load Model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStreamModel(seq_len=SEQ_LEN).to(DEVICE)
    model.load_state_dict(torch.load(r'c:\Users\SWH\Desktop\智能医疗检测系统\核心程序\deep_learning\final_model.pth', map_location=DEVICE))
    model.eval()
    
    # Iterate and find best samples per group
    # Groups: Unique (Size, Depth) pairs
    # Since iterating dataset is slow, we will collect predictions first
    
    results = []
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Batch 1 to keep index simple
    
    with torch.no_grad():
        for i, (data, avg_intensity, target_prob, target_size, target_depth) in enumerate(val_loader):
            if target_prob.item() < 0.5: continue
            
            data = data.to(DEVICE)
            avg_intensity = avg_intensity.to(DEVICE)
            
            pred_prob, pred_size, pred_depth = model(data, avg_intensity)
            
            t_size = target_size.item()
            t_depth = target_depth.item()
            p_size = pred_size.item()
            p_depth = pred_depth.item()
            
            # Error metric: Combined normalized error
            err_size = abs(p_size - t_size)
            err_depth = abs(p_depth - t_depth)
            total_err = err_size + err_depth
            
            results.append({
                'idx': i,
                'data': data.cpu().numpy(), # (1, Seq, 1, 12, 8)
                't_size': t_size,
                't_depth': t_depth,
                'p_size': p_size,
                'p_depth': p_depth,
                'err': total_err
            })
            
    # Group by (Size, Depth)
    df_res = pd.DataFrame(results)
    groups = df_res.groupby(['t_size', 't_depth'])
    
    for (size, depth), group in groups:
        # Get Top 3 best predictions
        best_samples = group.nsmallest(3, 'err')
        
        group_dir = os.path.join(SAMPLES_DIR, f'Size_{size}_Depth_{depth}')
        os.makedirs(group_dir, exist_ok=True)
        
        # Create summary figure for this group
        fig = plt.figure(figsize=(20, 12))
        plt.suptitle(f'Best Predictions for Size={size}cm, Depth={depth}cm', fontsize=16)
        
        for rank, (_, row) in enumerate(best_samples.iterrows()):
            # Plot sequence
            seq_data = row['data'][0, :, 0, :, :] # (10, 12, 8)
            
            # Subplot row for this sample
            # 10 frames + 1 text info
            for t in range(10):
                ax = plt.subplot(3, 11, rank * 11 + t + 1)
                ax.imshow(seq_data[t], cmap='turbo', vmin=0, vmax=1)
                ax.axis('off')
                if t == 0:
                    ax.set_ylabel(f'Sample {rank+1}', fontsize=12)
                    
            # Text info
            ax_text = plt.subplot(3, 11, rank * 11 + 11)
            text_str = (f"Pred Size: {row['p_size']:.3f}cm\n"
                        f"Err: {abs(row['p_size']-size):.3f}\n\n"
                        f"Pred Depth: {row['p_depth']:.3f}cm\n"
                        f"Err: {abs(row['p_depth']-depth):.3f}")
            ax_text.text(0.1, 0.5, text_str, va='center', fontsize=11)
            ax_text.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(group_dir, 'summary_best_3.png'), dpi=300)
        plt.close()
        
    print(f"Visualizations generated in {BASE_OUTPUT_DIR}")

if __name__ == '__main__':
    generate_paper_visualizations()
