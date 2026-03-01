import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStream3DCNNLSTM(nn.Module):
    def __init__(self, seq_len: int = 10, lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.stats_fc = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        fusion_in = 32 + lstm_hidden * 2
        self.fusion_fc = nn.Sequential(nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(dropout))

        self.prob_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.size_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.depth_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, stats_seq):
        x = x.permute(0, 2, 1, 3, 4)

        x_shape = F.relu(self.bn1(self.conv1(x)))
        x_shape = self.pool1(x_shape)
        x_shape = F.relu(self.bn2(self.conv2(x_shape)))
        x_shape = self.pool2(x_shape)

        b, c, t, h, w = x_shape.shape
        x_shape = x_shape.reshape(b, c * t, h, w)
        x_shape = self.spatial_conv(x_shape)
        x_shape = self.global_pool(x_shape)
        x_shape = x_shape.view(x_shape.size(0), -1)

        stats_embed = self.stats_fc(stats_seq)
        lstm_out, _ = self.lstm(stats_embed)
        lstm_feat = torch.mean(lstm_out, dim=1)

        fused = torch.cat([x_shape, lstm_feat], dim=1)
        features = self.fusion_fc(fused)

        prob = self.prob_head(features)
        size = self.size_head(features)
        depth = self.depth_head(features)
        return prob, size, depth


class DualStream3DCNNNoLSTM(nn.Module):
    def __init__(self, seq_len: int = 10, dropout: float = 0.3):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.stats_fc = nn.Sequential(nn.Linear(3, 16), nn.ReLU())

        fusion_in = 32 + 16
        self.fusion_fc = nn.Sequential(nn.Linear(fusion_in, 64), nn.ReLU(), nn.Dropout(dropout))

        self.prob_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.size_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.depth_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, stats_seq):
        x = x.permute(0, 2, 1, 3, 4)

        x_shape = F.relu(self.bn1(self.conv1(x)))
        x_shape = self.pool1(x_shape)
        x_shape = F.relu(self.bn2(self.conv2(x_shape)))
        x_shape = self.pool2(x_shape)

        b, c, t, h, w = x_shape.shape
        x_shape = x_shape.reshape(b, c * t, h, w)
        x_shape = self.spatial_conv(x_shape)
        x_shape = self.global_pool(x_shape)
        x_shape = x_shape.view(x_shape.size(0), -1)

        stats_embed = self.stats_fc(stats_seq)
        stats_feat = torch.mean(stats_embed, dim=1)

        fused = torch.cat([x_shape, stats_feat], dim=1)
        features = self.fusion_fc(fused)

        prob = self.prob_head(features)
        size = self.size_head(features)
        depth = self.depth_head(features)
        return prob, size, depth
