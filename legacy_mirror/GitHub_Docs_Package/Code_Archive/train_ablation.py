import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
import pandas as pd
import time
import copy
from final_model import DualStreamModel

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cleaned_Dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 15 # Reduced for demonstration, but enough for convergence on this data
SEQ_LEN = 10

class NpyDataset(Dataset):
    def __init__(self, sequences_path, labels_path):
        self.sequences = np.load(sequences_path)
        self.labels = np.load(labels_path) # [Prob, Size, Depth]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Seq: (10, 12, 8) -> Need (10, 1, 12, 8) for model input (B, Seq, 1, H, W)
        seq = torch.from_numpy(self.sequences[idx]).unsqueeze(1) # (10, 1, 12, 8)
        
        # Intensity Stats (Mean, Max, Std) calculation on the fly
        # seq is (1, 10, 12, 8)
        # Intensity stream needs (Mean, Max, Std) of the whole sequence pressure
        # Flatten to (10*12*8)
        flat_seq = seq.view(-1)
        mean_val = torch.mean(flat_seq)
        max_val = torch.max(flat_seq)
        std_val = torch.std(flat_seq)
        intensity_stats = torch.tensor([mean_val, max_val, std_val])
        
        label = torch.from_numpy(self.labels[idx])
        
        return seq, intensity_stats, label

def load_datasets():
    print("Loading datasets from .npy files...")
    train_ds = NpyDataset(
        os.path.join(DATA_DIR, "Train_File1_sequences.npy"),
        os.path.join(DATA_DIR, "Train_File1_labels.npy")
    )
    val_ds = NpyDataset(
        os.path.join(DATA_DIR, "Val_File2_sequences.npy"),
        os.path.join(DATA_DIR, "Val_File2_labels.npy")
    )
    test_ds = NpyDataset(
        os.path.join(DATA_DIR, "Test_File3_sequences.npy"),
        os.path.join(DATA_DIR, "Test_File3_labels.npy")
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds

def evaluate_model(model, test_loader, ablation_mode='DualStream'):
    model.eval()
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.MSELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_loss = 0.0
    mae_size = 0.0
    mae_depth = 0.0
    pos_count = 0
    
    with torch.no_grad():
        for seq, intensity, label in test_loader:
            seq, intensity, label = seq.to(device), intensity.to(device), label.to(device)
            
            if ablation_mode == 'ShapeOnly':
                intensity = torch.zeros_like(intensity)
            elif ablation_mode == 'IntensityOnly':
                seq = torch.zeros_like(seq)
            
            prob_pred, size_pred, depth_pred = model(seq, intensity)
            
            prob_target = label[:, 0].unsqueeze(1)
            size_target = label[:, 1].unsqueeze(1)
            depth_target = label[:, 2].unsqueeze(1)
            
            loss_prob = criterion_cls(prob_pred, (prob_target > 0.5).float())
            
            mask = prob_target > 0.5
            if mask.sum() > 0:
                loss_size = criterion_reg(size_pred[mask], size_target[mask])
                loss_depth = criterion_reg(depth_pred[mask], depth_target[mask])
                
                mae_size += torch.sum(torch.abs(size_pred[mask] - size_target[mask])).item()
                mae_depth += torch.sum(torch.abs(depth_pred[mask] - depth_target[mask])).item()
                pos_count += mask.sum().item()
            else:
                loss_size = torch.tensor(0.0).to(device)
                loss_depth = torch.tensor(0.0).to(device)
                
            loss = loss_prob + loss_size + loss_depth
            test_loss += loss.item() * seq.size(0)
    
    test_loss /= len(test_loader.dataset)
    if pos_count > 0:
        mae_size /= pos_count
        mae_depth /= pos_count
        
    return test_loss, mae_size, mae_depth

def train_model(model, train_loader, val_loader, ablation_mode='DualStream', epochs=10, name="Model"):
    print(f"\nStarting training for {name} (Mode: {ablation_mode})...")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.MSELoss() # Or L1Loss
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = []
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for seq, intensity, label in train_loader:
            seq, intensity, label = seq.to(device), intensity.to(device), label.to(device)
            
            # Ablation Logic
            if ablation_mode == 'ShapeOnly':
                intensity = torch.zeros_like(intensity)
            elif ablation_mode == 'IntensityOnly':
                seq = torch.zeros_like(seq)
                
            optimizer.zero_grad()
            
            prob_pred, size_pred, depth_pred = model(seq, intensity)
            
            # Labels
            prob_target = label[:, 0].unsqueeze(1)
            size_target = label[:, 1].unsqueeze(1)
            depth_target = label[:, 2].unsqueeze(1)
            
            # Loss
            loss_prob = criterion_cls(prob_pred, (prob_target > 0.5).float())
            
            # Only compute regression loss for positive samples
            mask = prob_target > 0.5
            if mask.sum() > 0:
                loss_size = criterion_reg(size_pred[mask], size_target[mask])
                loss_depth = criterion_reg(depth_pred[mask], depth_target[mask])
            else:
                loss_size = torch.tensor(0.0).to(device)
                loss_depth = torch.tensor(0.0).to(device)
                
            loss = loss_prob + loss_size + loss_depth
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * seq.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae_size = 0.0
        val_mae_depth = 0.0
        val_pos_count = 0
        
        with torch.no_grad():
            for seq, intensity, label in val_loader:
                seq, intensity, label = seq.to(device), intensity.to(device), label.to(device)
                
                if ablation_mode == 'ShapeOnly':
                    intensity = torch.zeros_like(intensity)
                elif ablation_mode == 'IntensityOnly':
                    seq = torch.zeros_like(seq)
                
                prob_pred, size_pred, depth_pred = model(seq, intensity)
                
                prob_target = label[:, 0].unsqueeze(1)
                size_target = label[:, 1].unsqueeze(1)
                depth_target = label[:, 2].unsqueeze(1)
                
                loss_prob = criterion_cls(prob_pred, (prob_target > 0.5).float())
                
                mask = prob_target > 0.5
                if mask.sum() > 0:
                    loss_size = criterion_reg(size_pred[mask], size_target[mask])
                    loss_depth = criterion_reg(depth_pred[mask], depth_target[mask])
                    
                    val_mae_size += torch.sum(torch.abs(size_pred[mask] - size_target[mask])).item()
                    val_mae_depth += torch.sum(torch.abs(depth_pred[mask] - depth_target[mask])).item()
                    val_pos_count += mask.sum().item()
                else:
                    loss_size = torch.tensor(0.0).to(device)
                    loss_depth = torch.tensor(0.0).to(device)
                    
                loss = loss_prob + loss_size + loss_depth
                val_loss += loss.item() * seq.size(0)
        
        val_loss /= len(val_loader.dataset)
        if val_pos_count > 0:
            val_mae_size /= val_pos_count
            val_mae_depth /= val_pos_count
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE Size: {val_mae_size:.4f} | Val MAE Depth: {val_mae_depth:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'model': name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae_size': val_mae_size,
            'val_mae_depth': val_mae_depth
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{name}.pth"))
    return model, history

def main():
    train_ds, val_ds, test_ds = load_datasets()
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False) # Only for final evaluation!
    
    # Combined dataset for Active Learning (Phase 3)
    mixed_ds = ConcatDataset([train_ds, val_ds])
    
    # Split mixed_ds into train and validation for Active Learning
    # to avoid using test_ds for validation
    mixed_val_size = int(len(mixed_ds) * 0.1) # 10% for validation
    mixed_train_size = len(mixed_ds) - mixed_val_size
    mixed_train_ds, mixed_val_ds = random_split(mixed_ds, [mixed_train_size, mixed_val_size])
    
    mixed_loader = DataLoader(mixed_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    mixed_val_loader = DataLoader(mixed_val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_history = []
    final_test_metrics = []
    
    # 1. Baseline Training (DualStream on File 1)
    print("\n=== Experiment 1: Baseline (DualStream) ===")
    model_baseline = DualStreamModel(seq_len=SEQ_LEN)
    model_baseline, hist_baseline = train_model(model_baseline, train_loader, val_loader, ablation_mode='DualStream', epochs=EPOCHS, name="Baseline_DualStream")
    all_history.extend(hist_baseline)
    # Evaluate on Test Set (File 3)
    test_loss, mae_size, mae_depth = evaluate_model(model_baseline, test_loader, ablation_mode='DualStream')
    final_test_metrics.append({'model': 'Baseline', 'test_loss': test_loss, 'mae_size': mae_size, 'mae_depth': mae_depth})
    
    # 2. Active Learning / Mixed Training (DualStream on File 1 + File 2)
    print("\n=== Experiment 2: Active Learning (Mixed Retraining) ===")
    # Load pretrained baseline weights
    model_active = DualStreamModel(seq_len=SEQ_LEN)
    model_active.load_state_dict(torch.load(os.path.join(MODEL_DIR, "Baseline_DualStream.pth")))
    # Retrain on mixed dataset (Validating on mixed_val_ds, NOT File 3)
    model_active, hist_active = train_model(model_active, mixed_loader, mixed_val_loader, ablation_mode='DualStream', epochs=EPOCHS, name="Active_DualStream") 
    all_history.extend(hist_active)
    # Evaluate on Test Set (File 3)
    test_loss, mae_size, mae_depth = evaluate_model(model_active, test_loader, ablation_mode='DualStream')
    final_test_metrics.append({'model': 'Active Learning', 'test_loss': test_loss, 'mae_size': mae_size, 'mae_depth': mae_depth})
    
    # 3. Ablation: Shape Only (on Baseline data for fair comparison)
    print("\n=== Experiment 3: Ablation (Shape Stream Only) ===")
    model_shape = DualStreamModel(seq_len=SEQ_LEN)
    model_shape, hist_shape = train_model(model_shape, train_loader, val_loader, ablation_mode='ShapeOnly', epochs=EPOCHS, name="Ablation_ShapeOnly")
    all_history.extend(hist_shape)
    # Evaluate on Test Set (File 3)
    test_loss, mae_size, mae_depth = evaluate_model(model_shape, test_loader, ablation_mode='ShapeOnly')
    final_test_metrics.append({'model': 'Shape Only', 'test_loss': test_loss, 'mae_size': mae_size, 'mae_depth': mae_depth})
    
    # 4. Ablation: Intensity Only (on Baseline data)
    print("\n=== Experiment 4: Ablation (Intensity Stream Only) ===")
    model_intensity = DualStreamModel(seq_len=SEQ_LEN)
    model_intensity, hist_intensity = train_model(model_intensity, train_loader, val_loader, ablation_mode='IntensityOnly', epochs=EPOCHS, name="Ablation_IntensityOnly")
    all_history.extend(hist_intensity)
    # Evaluate on Test Set (File 3)
    test_loss, mae_size, mae_depth = evaluate_model(model_intensity, test_loader, ablation_mode='IntensityOnly')
    final_test_metrics.append({'model': 'Intensity Only', 'test_loss': test_loss, 'mae_size': mae_size, 'mae_depth': mae_depth})
    
    # Save Logs
    df = pd.DataFrame(all_history)
    df.to_csv(os.path.join(OUTPUT_DIR, "training_logs_ablation.csv"), index=False)
    
    # Save Final Test Metrics
    df_test = pd.DataFrame(final_test_metrics)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "final_test_results.csv"), index=False)
    
    print(f"\nTraining logs saved to {os.path.join(OUTPUT_DIR, 'training_logs_ablation.csv')}")
    print(f"Final Test Metrics saved to {os.path.join(OUTPUT_DIR, 'final_test_results.csv')}")
    print("\n=== Final Test Set Results ===")
    print(df_test)

if __name__ == "__main__":
    main()
