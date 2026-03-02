
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from dataset import KeyframeDataset
from model import KeyframeDetector

def train():
    # Config
    DATA_DIR = r"c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\数据集 最新\建表数据"
    LABEL_FILE = r"c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\manual_keyframe_labels.json"
    MODEL_PATH = r"c:\Users\SWH\Desktop\智能医疗检测系统\核心程序\deep_learning\keyframe_model.pth"
    
    BATCH_SIZE = 16
    EPOCHS = 15 # Train longer
    LR = 0.001
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    full_dataset = KeyframeDataset(DATA_DIR, LABEL_FILE)
    
    # Split Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = KeyframeDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    # Metrics tracking
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for frames, meta, labels in train_loader:
            frames, meta, labels = frames.to(device), meta.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(frames, meta)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, meta, labels in val_loader:
                frames, meta, labels = frames.to(device), meta.to(device), labels.to(device)
                outputs, _ = model(frames, meta)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"Training complete in {time.time() - start_time:.1f}s")
    
    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Plot history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.savefig(r"c:\Users\SWH\Desktop\智能医疗检测系统\实验数据\training_history.png")
    print("History plot saved.")

if __name__ == "__main__":
    train()
