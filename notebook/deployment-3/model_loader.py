"""
Model loader for deployment inference.
Loads the trained multi-branch LSTM model and configuration.

Architecture matches LSTM.ipynb exactly:
- 9 branches: RawTime, RawFreq, RmsLms, WienerTD, WienerFFT, IMU,
              SpectralRaw, SpectralWiener, WindowStats
- 2-layer unidirectional LSTM, mean pooling, linear classifier
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Branch modules (match LSTM.ipynb exactly)
# ---------------------------------------------------------------------------

class UltraCompactBranch(nn.Module):
    """3-conv + pool + projection branch for 1-channel time-domain signals."""

    def __init__(self, in_channels: int = 1, d_model: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(32, d_model)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels, length)  or  (batch, channels, length)
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
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.proj(x))

        if reshape:
            x = x.view(batch_size, seq_len, -1)
        return x


class RawEMGTimeBranch(UltraCompactBranch):
    def __init__(self, d_model: int = 16):
        super().__init__(in_channels=1, d_model=d_model)


class RawEMGFreqBranch(nn.Module):
    """Raw EMG frequency-domain branch — computes FFT internally."""

    def __init__(self, d_model: int = 16, fft_bins: int = 64):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1, window_size)
        if x.dim() == 4:
            batch_size, seq_len, channels, length = x.shape
            x = x.view(batch_size * seq_len, channels, length)
            reshape = True
        else:
            reshape = False
            batch_size = x.shape[0]
            seq_len = 1
            x = x.unsqueeze(0) if x.dim() == 2 else x

        x_signal = x.squeeze(1)                           # (B, L)
        x_fft_mag = torch.abs(torch.fft.rfft(x_signal, dim=1))  # (B, L//2+1)

        if x_fft_mag.shape[1] != self.fft_bins:
            x_fft_mag = torch.nn.functional.interpolate(
                x_fft_mag.unsqueeze(1),
                size=self.fft_bins,
                mode='linear',
                align_corners=False,
            ).squeeze(1)

        x_fft_mag = x_fft_mag.unsqueeze(1)               # (B, 1, fft_bins)
        x = self.relu(self.bn1(self.conv1(x_fft_mag)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.proj(x))

        if reshape:
            x = x.view(batch_size, seq_len, -1)
        return x


class RMSLMSBranch(UltraCompactBranch):
    """RMS envelope branch — single time-domain channel."""
    def __init__(self, d_model: int = 16):
        super().__init__(in_channels=1, d_model=d_model)


class WienerTDBranch(UltraCompactBranch):
    """Wiener-filtered time-domain branch."""
    def __init__(self, d_model: int = 16):
        super().__init__(in_channels=1, d_model=d_model)


class WienerFFTBranch(nn.Module):
    """Wiener pre-computed FFT bins branch."""

    def __init__(self, d_model: int = 16, fft_bins: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(16, d_model)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 64)  or  (batch, 64)
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
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.proj(x))

        if reshape:
            x = x.view(batch_size, seq_len, -1)
        return x


class IMUBranch(UltraCompactBranch):
    """6-channel IMU branch."""
    def __init__(self, d_model: int = 16):
        super().__init__(in_channels=6, d_model=d_model)


class SpectralFeatureEncoder(nn.Module):
    """Lightweight MLP for 5-d spectral feature vectors."""

    def __init__(self, input_dim: int = 5, d_model: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 5)
        batch, seq_len, feat = x.shape
        x = self.net(x.view(batch * seq_len, feat))
        return x.view(batch, seq_len, -1)


class WindowStatsEncoder(nn.Module):
    """MLP for 16-d time-domain statistics (raw + wiener)."""

    def __init__(self, input_dim: int = 16, d_model: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 16)
        batch, seq_len, feat = x.shape
        x = self.net(x.view(batch * seq_len, feat))
        return x.view(batch, seq_len, -1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class EMGLSTMModel(nn.Module):
    """
    Multi-branch LSTM model matching LSTM.ipynb exactly.

    Input batch keys:
        raw          (batch, seq_len, 1, window_size)
        rms_lms      (batch, seq_len, 1, window_size)
        wiener_td    (batch, seq_len, 1, window_size)
        wiener_fft   (batch, seq_len, 64)
        imu          (batch, seq_len, 6, n_imu_samples)
        spectral_raw    (batch, seq_len, 5)
        spectral_wiener (batch, seq_len, 5)
        window_stats    (batch, seq_len, 16)
    """

    def __init__(
        self,
        num_classes: int = 2,
        d_model: int = 16,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.65,
        sequence_length: int = 8,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Signal branches
        self.raw_time_branch = RawEMGTimeBranch(d_model=d_model)
        self.raw_freq_branch = RawEMGFreqBranch(d_model=d_model, fft_bins=64)
        self.rms_lms_branch = RMSLMSBranch(d_model=d_model)
        self.wiener_td_branch = WienerTDBranch(d_model=d_model)
        self.wiener_fft_branch = WienerFFTBranch(d_model=d_model, fft_bins=64)
        self.imu_branch = IMUBranch(d_model=d_model)

        # Scalar feature encoders
        self.spectral_raw_encoder = SpectralFeatureEncoder(input_dim=5, d_model=d_model)
        self.spectral_wiener_encoder = SpectralFeatureEncoder(input_dim=5, d_model=d_model)
        self.window_stats_encoder = WindowStatsEncoder(input_dim=16, d_model=d_model)

        # LSTM (9 branches × d_model)
        feature_dim = 9 * d_model
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        # Classification head
        self.dropout_cls = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raw = batch["raw"]
        rms_lms = batch["rms_lms"]
        wiener_td = batch["wiener_td"]
        wiener_fft = batch["wiener_fft"]
        imu = batch["imu"]
        spectral_raw = batch["spectral_raw"]
        spectral_wiener = batch["spectral_wiener"]
        window_stats = batch["window_stats"]

        combined = torch.cat(
            [
                self.raw_time_branch(raw),
                self.raw_freq_branch(raw),
                self.rms_lms_branch(rms_lms),
                self.wiener_td_branch(wiener_td),
                self.wiener_fft_branch(wiener_fft),
                self.imu_branch(imu),
                self.spectral_raw_encoder(spectral_raw),
                self.spectral_wiener_encoder(spectral_wiener),
                self.window_stats_encoder(window_stats),
            ],
            dim=2,
        )  # (batch, seq_len, 9*d_model)

        lstm_out, _ = self.lstm(combined)
        pooled = lstm_out.mean(dim=1)               # mean pooling over time
        logits = self.classifier(self.dropout_cls(pooled))
        return logits


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str, device: str = "cpu"
) -> Tuple[EMGLSTMModel, Dict[str, Any], Dict[str, Any]]:
    """
    Load the trained model, deployment configuration, and StandardScalers.

    Returns:
        model   – loaded EMGLSTMModel in eval mode
        config  – deployment_config.json contents (+ class_names key)
        scalers – dict of StandardScaler objects keyed by feature name
    """
    model_dir = Path(model_dir)

    config_path = model_dir / "deployment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    scaler_path = model_dir / "standard_scalers.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scalers not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    model_path = model_dir / "emg_lstm_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    raw_cfg = checkpoint.get("model_config", {})
    model_params = {
        "num_classes":    raw_cfg.get("num_classes",    2),
        "d_model":        raw_cfg.get("d_model",        16),
        "hidden_size":    raw_cfg.get("hidden_size",    32),
        "num_layers":     raw_cfg.get("num_layers",     2),
        "dropout":        raw_cfg.get("dropout",        0.65),
        "sequence_length": raw_cfg.get("sequence_length", config.get("sequence_length", 8)),
        "bidirectional":  raw_cfg.get("bidirectional",  False),
    }

    model = EMGLSTMModel(**model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Normalise class names
    raw_names = checkpoint.get("class_names", None)
    if raw_names is None:
        class_names = ["REST", "FIST"]
    elif isinstance(raw_names, list) and all(isinstance(n, str) for n in raw_names):
        class_names = raw_names
    else:
        class_names = ["REST", "FIST"]

    config["class_names"] = class_names
    config["model_config"] = model_params

    return model, config, scalers
