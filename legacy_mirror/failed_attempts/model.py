
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # x: (Batch, C, H, W)
        att = torch.sigmoid(self.conv(x))
        return x * att

class KeyframeDetector(nn.Module):
    def __init__(self, sequence_length=30, input_h=60, input_w=40, 
                 use_attention=True, use_metadata=True, use_lstm=True):
        super(KeyframeDetector, self).__init__()
        
        self.use_attention = use_attention
        self.use_metadata = use_metadata
        self.use_lstm = use_lstm
        
        # 1. CNN Encoder (Spatial Features)
        layers = [
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 30x20
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 15x10
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]
        
        if self.use_attention:
            layers.append(SpatialAttention(64))
            
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.cnn = nn.Sequential(*layers)
        
        # 2. Metadata Encoder
        if self.use_metadata:
            self.meta_fc = nn.Sequential(
                nn.Linear(2, 16), # Size, Depth
                nn.ReLU()
            )
            cnn_out_dim = 64
            meta_out_dim = 16
            lstm_input_dim = cnn_out_dim + meta_out_dim
        else:
            self.meta_fc = None
            lstm_input_dim = 64
        
        # 3. Temporal Encoder (LSTM) or Simple Pooling
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
            self.attention_fc = nn.Linear(128, 1) # 128 because bidirectional (64*2)
            classifier_input_dim = 128
        else:
            # If no LSTM, we just average over time
            self.lstm = None
            self.attention_fc = None
            classifier_input_dim = lstm_input_dim
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, frames, metadata):
        # frames: (Batch, Seq_Len, H, W) -> unsqueeze channel -> (Batch*Seq, 1, H, W)
        batch_size, seq_len, h, w = frames.size()
        
        c_in = frames.view(batch_size * seq_len, 1, h, w)
        
        # CNN Feature Extraction
        cnn_out = self.cnn(c_in) # (Batch*Seq, 64, 1, 1)
        cnn_out = cnn_out.view(batch_size, seq_len, -1) # (Batch, Seq, 64)
        
        # Metadata Processing
        if self.use_metadata:
            # metadata: (Batch, 2) -> (Batch, 16) -> repeat for seq -> (Batch, Seq, 16)
            meta_out = self.meta_fc(metadata)
            meta_out = meta_out.unsqueeze(1).repeat(1, seq_len, 1)
            # Concatenate
            features = torch.cat([cnn_out, meta_out], dim=2) # (Batch, Seq, 80)
        else:
            features = cnn_out
        
        # Temporal Processing
        att_weights = None
        
        if self.use_lstm:
            # LSTM
            lstm_out, _ = self.lstm(features) # (Batch, Seq, 128)
            
            # Temporal Attention
            # Calculate attention weights for each time step
            att_weights = torch.softmax(self.attention_fc(lstm_out), dim=1) # (Batch, Seq, 1)
            
            # Weighted sum of LSTM outputs
            context_vector = torch.sum(lstm_out * att_weights, dim=1) # (Batch, 128)
        else:
            # Simple Global Average Pooling over time if no LSTM
            context_vector = torch.mean(features, dim=1)
            # Dummy attention weights for consistency
            att_weights = torch.ones(batch_size, seq_len, 1).to(frames.device) / seq_len
        
        # Classification
        prob = self.classifier(context_vector) # (Batch, 1)
        
        return prob, att_weights
