import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamModel(nn.Module):
    """
    Final optimized model for Nodule Detection, Size, and Depth Estimation.
    
    Architecture:
    - Stream 1 (Shape): 3D CNN taking normalized sequence (B, 1, Seq, 12, 8).
      Captures Spatio-Temporal shape features (sharpness, spread, evolution).
    - Stream 2 (Intensity): MLP taking average intensity (scalar).
      Captures absolute signal strength.
    - Fusion: Concatenates streams and uses specialized heads for Prob, Size, Depth.
    """
    def __init__(self, seq_len=10):
        super(DualStreamModel, self).__init__()
        
        # Branch 1: Shape Stream (3D CNN)
        # Input: (B, 1, Seq, 12, 8)
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) # -> (16, 5, 6, 4)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        # Output after global pool -> (32)
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Branch 2: Intensity Stream (MLP)
        self.intensity_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Shared Fusion Layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task Heads
        
        # 1. Detection Head (Probability)
        self.prob_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 2. Size Head (Regression)
        self.size_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 3. Depth Head (Regression)
        self.depth_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, avg_intensity):
        # x shape: (B, Seq, 1, 12, 8) -> Permute to (B, 1, Seq, 12, 8) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        # Branch 1: Shape
        x_shape = F.relu(self.bn1(self.conv1(x)))
        x_shape = self.pool1(x_shape)
        x_shape = F.relu(self.bn2(self.conv2(x_shape)))
        x_shape = self.global_pool(x_shape)
        x_shape = x_shape.view(x_shape.size(0), -1) # (B, 32)
        
        # Branch 2: Intensity
        if avg_intensity.dim() == 1:
            avg_intensity = avg_intensity.unsqueeze(1)
        x_intensity = self.intensity_mlp(avg_intensity) # (B, 32)
        
        # Fusion
        combined = torch.cat((x_shape, x_intensity), dim=1) # (B, 64)
        features = self.fusion_fc(combined)
        
        prob = self.prob_head(features)
        size = self.size_head(features)
        depth = self.depth_head(features)
        
        return prob, size, depth
