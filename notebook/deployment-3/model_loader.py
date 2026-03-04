"""
Model loader for deployment inference.
Loads the trained LSTM model and configuration.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class EMGLSTMModel(nn.Module):
    """
    Enhanced LSTM model for REST vs FIST classification with temporal attention.

    This matches the architecture defined in notebook/LSTM.ipynb, including:
    - CNN envelope branch
    - Minimal feature set per window (envelope + 3 bandpowers + 4 IMU stats)
    - Optional temporal feature expansion (rate-of-change, variance, trend)
    - Temporal attention over sequence timesteps
    """

    def __init__(
        self,
        num_classes: int = 2,
        window_size: int = 100,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.5,
        sequence_length: int = 16,
        bidirectional: bool = False,
        use_temporal_features: bool = True,
    ) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        self.window_size = window_size
        self.use_temporal_features = use_temporal_features

        # EMG envelope processor (simple CNN to extract amplitude patterns)
        self.emg_envelope_processor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 8),
        )

        # Base feature dimension per window:
        # - EMG envelope: 8 (from CNN)
        # - EMG bandpowers: 3 (low/mid/high)
        # - IMU: 4 (gyro_mag_mean, accel_mag_mean, gyro_mag_std, accel_mag_std)
        # - vs-REST features: 2 (envelope_vs_rest, high_bandpower_vs_rest)
        base_feature_dim = 8 + 3 + 4 + 2  # = 17 features per window

        # With temporal features: 17 * 4 = 68 features per window
        # (original + rate_of_change + variance + trend)
        if use_temporal_features:
            feature_dim = base_feature_dim * 4
        else:
            feature_dim = base_feature_dim

        # LSTM processes sequence
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # LSTM output dimension
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1),
        )

        # Classification head
        self.dropout_cls = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def _add_temporal_features(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Add temporal features to base per-window features:
        - Rate of change (derivative)
        - Temporal variance (consistency across sequence)
        - Trend (linear trend over timesteps)

        Args:
            combined_features: (batch, seq_len, feature_dim) - base features

        Returns:
            enhanced_features: (batch, seq_len, feature_dim * 4)
        """
        batch_size, seq_len, feature_dim = combined_features.shape

        # Rate of change between consecutive windows
        if seq_len > 1:
            rate_of_change = combined_features[:, 1:, :] - combined_features[:, :-1, :]
            rate_of_change = torch.cat(
                [torch.zeros_like(combined_features[:, :1, :]), rate_of_change],
                dim=1,
            )
        else:
            rate_of_change = torch.zeros_like(combined_features)

        # Temporal variance across sequence (same for all timesteps in a sequence)
        temporal_var = torch.std(combined_features, dim=1, keepdim=True)
        temporal_var = temporal_var.expand(-1, seq_len, -1)

        # Linear trend across sequence (slope per feature)
        if seq_len > 2:
            indices = torch.arange(
                seq_len, dtype=combined_features.dtype, device=combined_features.device
            )
            indices = indices.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
            mean_idx = (seq_len - 1) / 2.0
            indices_centered = indices - mean_idx

            mean_feat = torch.mean(combined_features, dim=1, keepdim=True)
            numerator = torch.sum(
                indices_centered * (combined_features - mean_feat),
                dim=1,
                keepdim=True,
            )
            denominator = torch.sum(indices_centered**2, dim=1, keepdim=True) + 1e-10
            trend = numerator / denominator  # (batch, 1, features)
            trend = trend.expand(-1, seq_len, -1)
        else:
            trend = torch.zeros_like(combined_features)

        # Concatenate temporal features
        enhanced_features = torch.cat(
            [combined_features, rate_of_change, temporal_var, trend],
            dim=2,
        )

        return enhanced_features

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch: Dictionary with:
                - 'emg_envelope': (batch, seq_len, window_size)
                - 'emg_bandpowers': (batch, seq_len, 3)
                - 'imu_features': (batch, seq_len, 4)
                - 'vs_rest': (batch, seq_len, 2) - [envelope_vs_rest, high_bandpower_vs_rest]

        Returns:
            logits: (batch, num_classes)
        """
        emg_envelope = batch["emg_envelope"]
        emg_bandpowers = batch["emg_bandpowers"]
        imu_features = batch["imu_features"]
        vs_rest = batch.get("vs_rest", torch.zeros(emg_envelope.shape[0], emg_envelope.shape[1], 2, device=emg_envelope.device))

        batch_size, seq_len = emg_envelope.shape[:2]

        # Process EMG envelope through CNN
        envelope_flat = emg_envelope.view(batch_size * seq_len, 1, self.window_size)
        envelope_features = self.emg_envelope_processor(envelope_flat)
        envelope_features = envelope_features.view(batch_size, seq_len, -1)

        # Concatenate base features (including vs-REST features)
        combined_features = torch.cat(
            [envelope_features, emg_bandpowers, imu_features, vs_rest],
            dim=2,
        )  # (batch, seq_len, 17)

        # Add temporal features if enabled
        if self.use_temporal_features:
            combined_features = self._add_temporal_features(combined_features)

        # LSTM processing
        lstm_out, _ = self.lstm(combined_features)  # (batch, seq_len, hidden)

        # Temporal attention pooling over timesteps
        attention_scores = self.temporal_attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        pooled = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)

        # Classification
        pooled = self.dropout_cls(pooled)
        logits = self.classifier(pooled)

        return logits


def load_model(model_dir: str, device: str = "cpu") -> Tuple[EMGLSTMModel, Dict[str, Any], Dict[str, Any]]:
    """
    Load the trained model, configuration, and scalers.
    
    Args:
        model_dir: Directory containing deployment files
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded PyTorch model
        config: Deployment configuration dictionary
        scalers: Dictionary of StandardScaler objects
    """
    model_dir = Path(model_dir)
    
    # Load configuration
    config_path = model_dir / 'deployment_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load scalers
    scaler_path = model_dir / 'standard_scalers.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    # Load model
    model_path = model_dir / 'emg_lstm_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # PyTorch 2.6+ defaults to weights_only=True for security, but our checkpoint
    # contains sklearn LabelEncoder objects. Since this is our own trusted model file,
    # we safely disable weights_only restriction.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model config - filter to only include valid parameters
    raw_model_config = checkpoint.get('model_config', {})
    
    # Define valid parameters for EMGLSTMModel
    valid_params = {
        "num_classes": raw_model_config.get("num_classes", 2),
        "window_size": raw_model_config.get("window_size", config.get("window_size", 100)),
        "hidden_size": raw_model_config.get("hidden_size", 32),
        "num_layers": raw_model_config.get("num_layers", 2),
        "dropout": raw_model_config.get("dropout", 0.5),
        "sequence_length": raw_model_config.get("sequence_length", config.get("sequence_length", 16)),
        "bidirectional": raw_model_config.get("bidirectional", False),
        "use_temporal_features": raw_model_config.get("use_temporal_features", True),
    }
    
    # Create model instance with only valid parameters
    model = EMGLSTMModel(**valid_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract class names - handle both string and numeric formats
    class_names_raw = checkpoint.get('class_names', None)
    if class_names_raw is None:
        class_names = ['REST', 'FIST']
    elif isinstance(class_names_raw, list):
        # Check if it's numeric [0, 1] or string ['REST', 'FIST']
        if len(class_names_raw) == 2 and all(isinstance(x, (int, float)) for x in class_names_raw):
            # Convert numeric to string labels (assume 0=REST, 1=FIST)
            class_names = ['REST', 'FIST']
        elif all(isinstance(x, str) for x in class_names_raw):
            class_names = class_names_raw
        else:
            class_names = ['REST', 'FIST']
    else:
        class_names = ['REST', 'FIST']
    
    config['class_names'] = class_names
    config['model_config'] = valid_params  # Store filtered config
    
    return model, config, scalers

