
import torch
import numpy as np
from scipy import ndimage
from collections import deque
import sys
import os

# Add deep_learning directory to path to import model
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '核心程序', 'deep_learning'))
from model import KeyframeDetector

class RealtimeNoduleDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = KeyframeDetector().to(self.device)
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Deep Learning Model loaded successfully.")
        except Exception as e:
            print(f"Error loading DL model: {e}")
            self.model = None
            
        # Buffer for sequence data (need 30 frames for LSTM)
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Interpolation factor (must match training)
        self.interp_factor = 5
        
    def process_frame(self, frame_flat, size_mm=10.0, depth_mm=10.0):
        """
        Process a single new frame.
        Returns: probability (float) or None if buffer not full
        """
        if self.model is None:
            return 0.0
            
        # 1. Preprocess Frame
        try:
            mat = frame_flat.reshape(12, 8)
            mn, mx = mat.min(), mat.max()
            if mx - mn > 1e-6:
                norm = (mat - mn) / (mx - mn)
            else:
                norm = mat - mn
            
            # Interpolate
            img = ndimage.zoom(norm, self.interp_factor, order=1) # 12x8 -> 60x40
            
            # Add to buffer
            self.frame_buffer.append(img)
            
            # 2. Check Buffer
            if len(self.frame_buffer) < self.sequence_length:
                return 0.0
            
            # 3. Prepare Batch
            # (1, 30, 60, 40)
            seq_array = np.array(self.frame_buffer)
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Metadata
            meta_tensor = torch.tensor([[size_mm, depth_mm]], dtype=torch.float32).to(self.device)
            
            # 4. Inference
            with torch.no_grad():
                prob, _ = self.model(seq_tensor, meta_tensor)
                return prob.item()
                
        except Exception as e:
            print(f"DL Inference Error: {e}")
            return 0.0

    def predict_sequence(self, sequence_data, size_mm=10.0, depth_mm=10.0):
        """
        Directly predict probability for a given sequence of raw frames.
        sequence_data: numpy array of shape (30, 96) or similar
        """
        if self.model is None:
            return 0.0
            
        try:
            # Check length
            if len(sequence_data) != self.sequence_length:
                # Pad if short? Or just return 0
                if len(sequence_data) < self.sequence_length:
                    # Pad with first frame
                    pad_len = self.sequence_length - len(sequence_data)
                    pad = np.repeat(sequence_data[[0]], pad_len, axis=0)
                    sequence_data = np.vstack([pad, sequence_data])
                else:
                    sequence_data = sequence_data[-self.sequence_length:]
            
            # Preprocess Batch
            processed_seq = []
            for frame_flat in sequence_data:
                mat = frame_flat.reshape(12, 8)
                mn, mx = mat.min(), mat.max()
                if mx - mn > 1e-6:
                    norm = (mat - mn) / (mx - mn)
                else:
                    norm = mat - mn
                img = ndimage.zoom(norm, self.interp_factor, order=1)
                processed_seq.append(img)
            
            # To Tensor
            seq_array = np.array(processed_seq) # (30, 60, 40)
            seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            meta_tensor = torch.tensor([[size_mm, depth_mm]], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                prob, _ = self.model(seq_tensor, meta_tensor)
                return prob.item()
                
        except Exception as e:
            print(f"DL Sequence Prediction Error: {e}")
            return 0.0

    def analyze_size_depth_distribution(self, current_size_prior, current_depth_prior):
        """
        Analyze the probability distribution of size and depth.
        Vary size and depth around the prior (or across the full range)
        and see how the keyframe probability changes.
        """
        if self.model is None or len(self.frame_buffer) < self.sequence_length:
            return None, None

        try:
            # Prepare sequence tensor (Batch=1, Seq, H, W) -> will be repeated
            seq_array = np.array(self.frame_buffer)
            seq_tensor_base = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # --- Size Analysis ---
            # Vary size from 0.0 to 50.0 mm (0-5cm)
            sizes = np.linspace(0, 50, 51) # 1mm steps
            size_batch_meta = []
            for s in sizes:
                size_batch_meta.append([s, current_depth_prior]) # Fix depth, vary size
            
            size_meta_tensor = torch.tensor(size_batch_meta, dtype=torch.float32).to(self.device)
            # Repeat sequence for batch
            size_seq_tensor = seq_tensor_base.repeat(len(sizes), 1, 1, 1)
            
            with torch.no_grad():
                size_probs, _ = self.model(size_seq_tensor, size_meta_tensor)
                size_probs = size_probs.cpu().numpy().flatten()
            
            # --- Depth Analysis ---
            # Vary depth from 0.0 to 50.0 mm
            depths = np.linspace(0, 50, 51)
            depth_batch_meta = []
            for d in depths:
                depth_batch_meta.append([current_size_prior, d]) # Fix size, vary depth
            
            depth_meta_tensor = torch.tensor(depth_batch_meta, dtype=torch.float32).to(self.device)
            depth_seq_tensor = seq_tensor_base.repeat(len(depths), 1, 1, 1)
            
            with torch.no_grad():
                depth_probs, _ = self.model(depth_seq_tensor, depth_meta_tensor)
                depth_probs = depth_probs.cpu().numpy().flatten()
                
            return (sizes, size_probs), (depths, depth_probs)
            
        except Exception as e:
            print(f"Distribution Analysis Error: {e}")
            return None, None

    def reset(self):
        self.frame_buffer.clear()
