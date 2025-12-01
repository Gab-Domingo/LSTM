#!/usr/bin/env python3
"""
EMG Gesture Recognition Inference Script for Raspberry Pi
========================================================
Optimized for Raspberry Pi with simple terminal logging.
Only logs when gesture recognition changes.

Usage:
    python inference.py --stream  # For UDP streaming (via udp_receiver.py)
    python inference.py --raw <raw.csv> --rms <rms.csv> --wiener <wiener.csv> --imu <imu.csv>
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

# =============================================================
# MODEL ARCHITECTURE (must match training: d_model=16, hidden_size=32, num_layers=2)
# =============================================================

class UltraCompactBranch(nn.Module):
    """Ultra-compact branch: 2 conv layers + pooling"""
    def __init__(self, in_channels=1, d_model=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(16, d_model)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, seq_len, channels, length = x.shape
            x = x.view(batch_size * seq_len, channels, length)
            reshape = True
        else:
            reshape = False
            batch_size = x.shape[0]
            seq_len = 1

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.proj(x)
        x = self.dropout(x)

        if reshape:
            x = x.view(batch_size, seq_len, -1)
        return x

class RawEMGTimeBranch(UltraCompactBranch):
    def __init__(self, d_model=16):
        super().__init__(in_channels=1, d_model=d_model)

class RawEMGFreqBranch(nn.Module):
    def __init__(self, d_model=16, fft_bins=64):
        super().__init__()
        self.fft_bins = fft_bins
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(16, d_model)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, seq_len, channels, length = x.shape
            x = x.view(batch_size * seq_len, channels, length)
            reshape = True
        else:
            reshape = False
            batch_size = x.shape[0]
            seq_len = 1
            x = x.unsqueeze(0) if x.dim() == 2 else x

        x_signal = x.squeeze(1)
        x_fft = torch.fft.rfft(x_signal, dim=1)
        x_fft_mag = torch.abs(x_fft)

        if x_fft_mag.shape[1] != self.fft_bins:
            x_fft_mag = x_fft_mag.unsqueeze(1)
            x_fft_mag = torch.nn.functional.interpolate(
                x_fft_mag, size=self.fft_bins, mode='linear', align_corners=False
            )
            x_fft_mag = x_fft_mag.squeeze(1)

        x_fft_mag = x_fft_mag.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x_fft_mag)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.proj(x)
        x = self.dropout(x)

        if reshape:
            x = x.view(batch_size, seq_len, -1)
        return x

class RMSLMSBranch(UltraCompactBranch):
    def __init__(self, d_model=16):
        super().__init__(in_channels=1, d_model=d_model)

class WienerTDBranch(UltraCompactBranch):
    def __init__(self, d_model=16):
        super().__init__(in_channels=1, d_model=d_model)

class WienerFFTBranch(nn.Module):
    def __init__(self, d_model=16, fft_bins=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(16, d_model)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x = x.view(batch_size * seq_len, 1, features)
            reshape = True
        else:
            reshape = False
            batch_size = x.shape[0]
            seq_len = 1
            x = x.unsqueeze(1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.proj(x)
        x = self.dropout(x)

        if reshape:
            x = x.view(batch_size, seq_len, -1)
        return x

class IMUBranch(UltraCompactBranch):
    def __init__(self, d_model=16):
        super().__init__(in_channels=6, d_model=d_model)

class EMGLSTMModel(nn.Module):
    def __init__(self, num_classes=2, d_model=16, hidden_size=32, num_layers=2,
                 dropout=0.8, sequence_length=8, bidirectional=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

        self.raw_time_branch = RawEMGTimeBranch(d_model=d_model)
        self.raw_freq_branch = RawEMGFreqBranch(d_model=d_model, fft_bins=64)
        self.rms_lms_branch = RMSLMSBranch(d_model=d_model)
        self.wiener_td_branch = WienerTDBranch(d_model=d_model)
        self.wiener_fft_branch = WienerFFTBranch(d_model=d_model, fft_bins=64)
        self.imu_branch = IMUBranch(d_model=d_model)

        feature_dim = 6 * d_model
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.dropout_cls = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, batch):
        raw = batch['raw']
        rms_lms = batch['rms_lms']
        wiener_td = batch['wiener_td']
        wiener_fft = batch['wiener_fft']
        imu = batch['imu']

        batch_size, seq_len = raw.shape[:2]

        features_raw_time = self.raw_time_branch(raw)
        features_raw_freq = self.raw_freq_branch(raw)
        features_rms_lms = self.rms_lms_branch(rms_lms)
        features_wiener_td = self.wiener_td_branch(wiener_td)
        features_wiener_fft = self.wiener_fft_branch(wiener_fft)
        features_imu = self.imu_branch(imu)

        combined_features = torch.cat([
            features_raw_time,
            features_raw_freq,
            features_rms_lms,
            features_wiener_td,
            features_wiener_fft,
            features_imu
        ], dim=2)

        lstm_out, (h_n, c_n) = self.lstm(combined_features)
        pooled = lstm_out.mean(dim=1)
        pooled = self.dropout_cls(pooled)
        logits = self.classifier(pooled)

        return logits


# =============================================================
# PREPROCESSING UTILITIES (Two-Step Normalization)
# =============================================================

def normalize_window_per_user(signal_window: np.ndarray, method: str = 'z-score') -> np.ndarray:
    """
    STEP 1: Per-window z-score normalization (removes user-specific baselines/scales).
    """
    if method == 'z-score':
        mean = np.mean(signal_window)
        std = np.std(signal_window)
        if std < 1e-8:
            return signal_window - mean
        return (signal_window - mean) / std
    return signal_window


def compute_spectral_features(signal_window: np.ndarray, sampling_rate: int = 500) -> Dict:
    """Compute spectral features from a signal window"""
    from scipy.fft import fft, fftfreq
    
    n = len(signal_window)
    fft_vals = fft(signal_window)
    fft_magnitude = np.abs(fft_vals[:n//2])

    if len(fft_magnitude) >= 64:
        indices = np.linspace(0, len(fft_magnitude)-1, 64).astype(int)
        fft_magnitude_64 = fft_magnitude[indices]
    else:
        fft_magnitude_64 = np.interp(np.linspace(0, len(fft_magnitude)-1, 64),
                                     np.arange(len(fft_magnitude)), fft_magnitude)

    freqs = fftfreq(n, 1/sampling_rate)[:n//2]

    low_mask = (freqs >= 20) & (freqs <= 60)
    bandpower_low = np.sum(fft_magnitude[low_mask]**2) if np.any(low_mask) else 0.0

    mid_mask = (freqs >= 60) & (freqs <= 150)
    bandpower_mid = np.sum(fft_magnitude[mid_mask]**2) if np.any(mid_mask) else 0.0

    high_mask = (freqs >= 150) & (freqs <= 250)
    bandpower_high = np.sum(fft_magnitude[high_mask]**2) if np.any(high_mask) else 0.0

    power_spectrum = fft_magnitude**2
    power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
    spectral_entropy = -np.sum(power_spectrum_norm * np.log(power_spectrum_norm + 1e-10))

    if np.sum(power_spectrum) > 0:
        spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
    else:
        spectral_centroid = 0.0

    return {
        'fft_magnitude': fft_magnitude_64.astype(np.float32),
        'bandpower_low': np.float32(bandpower_low),
        'bandpower_mid': np.float32(bandpower_mid),
        'bandpower_high': np.float32(bandpower_high),
        'spectral_entropy': np.float32(spectral_entropy),
        'spectral_centroid': np.float32(spectral_centroid)
    }


def create_window_from_data(
    raw_signal: np.ndarray,
    rms_signal: np.ndarray,
    lms_signal: np.ndarray,
    wiener_td: np.ndarray,
    wiener_fft: Optional[np.ndarray],
    imu_data: np.ndarray,
    window_start_idx: int,
    window_size: int = 100,
    sampling_rate: int = 500,
    scalers: Optional[Dict] = None
) -> Dict:
    """
    Create a single window from raw data with two-step normalization.
    
    STEP 1: Per-window z-score normalization
    STEP 2: Global StandardScaler normalization (if scalers provided)
    """
    end_idx = window_start_idx + window_size
    
    # Extract windows
    raw_window = raw_signal[window_start_idx:end_idx].copy()
    
    # Align other signals
    if len(rms_signal) >= end_idx:
        rms_window = rms_signal[window_start_idx:end_idx]
        lms_window = lms_signal[window_start_idx:end_idx]
    else:
        rms_window = np.interp(np.arange(window_size), 
                               np.linspace(0, window_size-1, len(rms_signal)), 
                               rms_signal)
        lms_window = np.interp(np.arange(window_size),
                               np.linspace(0, window_size-1, len(lms_signal)),
                               lms_signal)
    
    if len(wiener_td) >= end_idx:
        wiener_td_window = wiener_td[window_start_idx:end_idx]
    else:
        wiener_td_window = np.interp(np.arange(window_size),
                                     np.linspace(0, window_size-1, len(wiener_td)),
                                     wiener_td)
    
    # Wiener FFT
    if wiener_fft is not None:
        wiener_fft = np.asarray(wiener_fft)
        if wiener_fft.ndim == 1:
            wiener_fft_window = wiener_fft.copy()
            if len(wiener_fft_window) != 64:
                if len(wiener_fft_window) > 64:
                    wiener_fft_window = wiener_fft_window[:64]
                else:
                    padded = np.zeros(64, dtype=wiener_fft_window.dtype)
                    padded[:len(wiener_fft_window)] = wiener_fft_window
                    wiener_fft_window = padded
        elif wiener_fft.ndim == 2:
            if len(wiener_fft) > window_start_idx:
                wiener_fft_window = wiener_fft[min(window_start_idx, len(wiener_fft)-1)]
            else:
                wiener_fft_window = wiener_fft[-1]
        else:
            spectral = compute_spectral_features(raw_window, sampling_rate)
            wiener_fft_window = spectral['fft_magnitude']
    else:
        spectral = compute_spectral_features(raw_window, sampling_rate)
        wiener_fft_window = spectral['fft_magnitude']
    
    # IMU window
    window_duration_ms = window_size / sampling_rate * 1000
    imu_samples_needed = int(window_duration_ms / 10)  # IMU at 100Hz
    if len(imu_data) >= imu_samples_needed:
        imu_window = imu_data[:imu_samples_needed]
    else:
        imu_window = np.zeros((imu_samples_needed, 6))
        imu_window[:len(imu_data)] = imu_data
    
    # STEP 1: Per-window z-score normalization
    raw_window = normalize_window_per_user(raw_window, method='z-score')
    rms_window = normalize_window_per_user(rms_window, method='z-score')
    lms_window = normalize_window_per_user(lms_window, method='z-score')
    wiener_td_window = normalize_window_per_user(wiener_td_window, method='z-score')
    if len(wiener_fft_window) > 0:
        wiener_fft_window = normalize_window_per_user(wiener_fft_window, method='z-score')
    
    # IMU: normalize each channel separately
    imu_window_normalized = np.zeros_like(imu_window)
    for ch in range(imu_window.shape[1]):
        imu_window_normalized[:, ch] = normalize_window_per_user(imu_window[:, ch], method='z-score')
    imu_window = imu_window_normalized.T  # Transpose to (6, samples)
    
    # STEP 2: Global StandardScaler normalization (if scalers provided)
    if scalers:
        raw_window = scalers['raw'].transform(raw_window.reshape(-1, 1)).flatten()
        rms_lms_flat = np.stack([rms_window, lms_window], axis=0).flatten()
        rms_lms_norm = scalers['rms_lms'].transform(rms_lms_flat.reshape(-1, 1)).flatten()
        rms_lms_norm = rms_lms_norm.reshape(2, -1)
        rms_window = rms_lms_norm[0]
        lms_window = rms_lms_norm[1]
        wiener_td_window = scalers['wiener_td'].transform(wiener_td_window.reshape(-1, 1)).flatten()
        wiener_fft_window = scalers['wiener_fft'].transform(wiener_fft_window.reshape(1, -1)).flatten()
        imu_flat = imu_window.flatten()
        imu_norm = scalers['imu'].transform(imu_flat.reshape(-1, 1)).flatten()
        imu_window = imu_norm.reshape(6, -1)
    
    return {
        'raw': raw_window.astype(np.float32),
        'rms_lms': np.stack([rms_window, lms_window], axis=0).astype(np.float32),
        'wiener_td': wiener_td_window.astype(np.float32),
        'wiener_fft': wiener_fft_window.astype(np.float32),
        'imu': imu_window.astype(np.float32)
    }


def create_sequence_from_windows(windows: List[Dict], sequence_length: int = 8) -> Dict:
    """Create a sequence from multiple windows for model input."""
    if len(windows) < sequence_length:
        last_window = windows[-1] if windows else None
        while len(windows) < sequence_length:
            if last_window:
                windows.append(last_window)
            else:
                windows.append({
                    'raw': np.zeros(100, dtype=np.float32),
                    'rms_lms': np.zeros((2, 100), dtype=np.float32),
                    'wiener_td': np.zeros(100, dtype=np.float32),
                    'wiener_fft': np.zeros(64, dtype=np.float32),
                    'imu': np.zeros((6, 10), dtype=np.float32)
                })
    
    windows = windows[-sequence_length:]
    
    raw_seq = [w['raw'] for w in windows]
    rms_lms_seq = [w['rms_lms'][0] for w in windows]  # Use RMS only
    wiener_td_seq = [w['wiener_td'] for w in windows]
    wiener_fft_seq = [w['wiener_fft'] for w in windows]
    imu_seq = [w['imu'] for w in windows]
    
    return {
        'raw': torch.FloatTensor(np.stack(raw_seq)).unsqueeze(1),
        'rms_lms': torch.FloatTensor(np.stack(rms_lms_seq)).unsqueeze(1),
        'wiener_td': torch.FloatTensor(np.stack(wiener_td_seq)).unsqueeze(1),
        'wiener_fft': torch.FloatTensor(np.stack(wiener_fft_seq)),
        'imu': torch.FloatTensor(np.stack(imu_seq))
    }


# =============================================================
# MODEL LOADER
# =============================================================

def load_model(model_path: Path, device: str = 'cpu') -> Tuple[nn.Module, Dict, List[str]]:
    """Load trained model from file."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    model = EMGLSTMModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    class_names = checkpoint['class_names']
    
    return model, config, class_names


def load_scalers(scaler_path: Path) -> Dict:
    """Load StandardScalers for two-step normalization."""
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    return scaler_data


# =============================================================
# INFERENCE FUNCTION
# =============================================================

def predict_gesture(
    model: nn.Module,
    sequence: Dict,
    threshold: float = 0.70,
    device: str = 'cpu'
) -> Tuple[int, float, np.ndarray]:
    """
    Predict gesture from a sequence.
    
    Returns:
        prediction: Predicted class (0=REST, 1=FIST)
        confidence: Confidence score (max probability)
        probabilities: Array of class probabilities
    """
    sequence = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in sequence.items()}
    
    batch = {k: v.unsqueeze(0) for k, v in sequence.items()}
    
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()[0]
    
    # Use threshold for safety-critical applications
    if probs_np[1] > threshold:
        prediction = 1  # FIST
    else:
        prediction = 0  # REST
    
    confidence = float(np.max(probs_np))
    
    return prediction, confidence, probs_np


# =============================================================
# SIMPLE TERMINAL LOGGER (only logs on gesture change)
# =============================================================

class SimpleLogger:
    """Simple terminal logger - updates on gesture change or periodically to show system is alive."""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Args:
            update_interval: Update display every N seconds even if gesture doesn't change (default: 1.0s)
        """
        self.last_gesture = None
        self.last_confidence = None
        self.last_probabilities = None
        self.count = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.update_interval = update_interval
        
    def log(self, gesture_name: str, confidence: float, probabilities: np.ndarray):
        """Log when gesture changes OR periodically to show system is working."""
        elapsed = time.time() - self.start_time
        time_since_update = time.time() - self.last_update_time
        
        # Update if gesture changed OR if enough time has passed (show system is alive)
        gesture_changed = gesture_name != self.last_gesture
        should_update_periodic = time_since_update >= self.update_interval
        
        if gesture_changed or should_update_periodic:
            # Use carriage return to update same line
            print(f"\r[{elapsed:7.1f}s] Gesture: {gesture_name:5s} | Confidence: {confidence:.2%} | "
                  f"REST: {probabilities[0]:.2%} | FIST: {probabilities[1]:.2%}     ", 
                  end='', flush=True)
            
            if gesture_changed:
                self.count += 1
            
            self.last_gesture = gesture_name
            self.last_confidence = confidence
            self.last_probabilities = probabilities
            self.last_update_time = time.time()


# =============================================================
# MAIN INFERENCE SCRIPT
# =============================================================

def main():
    parser = argparse.ArgumentParser(description='EMG Gesture Recognition Inference')
    parser.add_argument('--model', type=str, default='emg_lstm_model.pt',
                       help='Path to model file')
    parser.add_argument('--config', type=str, default='deployment_config.json',
                       help='Path to deployment config file')
    parser.add_argument('--scalers', type=str, default='standard_scalers.pkl',
                       help='Path to StandardScalers file')
    parser.add_argument('--raw', type=str, help='Path to raw EMG CSV file')
    parser.add_argument('--rms', type=str, help='Path to RMS/LMS CSV file')
    parser.add_argument('--wiener', type=str, help='Path to Wiener CSV file')
    parser.add_argument('--imu', type=str, help='Path to IMU CSV file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run inference on (use cpu for Raspberry Pi)')
    parser.add_argument('--window-start', type=int, default=0,
                       help='Starting index for window')
    
    args = parser.parse_args()
    
    # Load deployment configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    threshold = config.get('optimal_threshold', 0.70)
    window_size = config['window_size']
    sequence_length = config['sequence_length']
    
    # Load StandardScalers
    scaler_path = Path(args.scalers)
    scalers = None
    if scaler_path.exists():
        scalers = load_scalers(scaler_path)
        print(f"✓ StandardScalers loaded")
    else:
        print(f"⚠️  StandardScalers not found: {scaler_path}")
        print(f"   Inference will use STEP 1 normalization only")
    
    print(f"Loading model from: {args.model}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Window size: {window_size}")
    print(f"Sequence length: {sequence_length}")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    model, model_config, class_names = load_model(model_path, device=args.device)
    print(f"✓ Model loaded successfully")
    print(f"  Classes: {class_names}")
    print(f"  Parameters: {config['model_parameters']:,}")
    
    # Load data if CSV files provided
    if args.raw and args.rms and args.wiener and args.imu:
        print(f"\nLoading data from CSV files...")
        raw_df = pd.read_csv(args.raw)
        rms_df = pd.read_csv(args.rms)
        wiener_df = pd.read_csv(args.wiener)
        imu_df = pd.read_csv(args.imu)
        
        # Extract signals
        raw_signal = raw_df['ch1'].values
        rms_signal = rms_df['rms_ch1'].values
        lms_signal = rms_df['lms_ch1'].values
        wiener_td = wiener_df['wiener_td_ch1'].values
        
        # Extract FFT bins
        fft_cols = [f'fft_bin_{i}' for i in range(64)]
        if all(col in wiener_df.columns for col in fft_cols):
            wiener_fft = wiener_df[fft_cols].values
        else:
            wiener_fft = None
        
        imu_data = imu_df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
        
        # Create windows for sequence
        windows = []
        stride = config.get('stride', 25)
        for i in range(args.window_start, len(raw_signal) - window_size, stride):
            window = create_window_from_data(
                raw_signal, rms_signal, lms_signal, wiener_td, wiener_fft, imu_data,
                i, window_size, config['sampling_rate'], scalers
            )
            windows.append(window)
            
            if len(windows) >= sequence_length:
                break
        
        # Create sequence
        sequence = create_sequence_from_windows(windows, sequence_length)
        
        # Predict
        print(f"\nRunning inference...")
        prediction, confidence, probs = predict_gesture(model, sequence, threshold, args.device)
        
        print(f"\n{'='*60}")
        print(f"PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"Predicted class: {class_names[prediction]} (class {prediction})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities:")
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            print(f"  {name}: {prob:.4f}")
        print(f"{'='*60}")
    
    else:
        print("\n⚠️  No CSV files provided. Use --raw, --rms, --wiener, --imu to provide data.")
        print("For streaming inference, use udp_receiver.py")


if __name__ == '__main__':
    main()
