"""
Sliding window buffer for real-time data processing.
Maintains buffers for EMG and IMU data with proper alignment.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


# Feature keys produced by RealTimePreprocessor.normalize_features()
# and consumed by EMGLSTMModel.forward()
_MODEL_FEATURE_KEYS = (
    'raw',
    'rms_lms',
    'wiener_td',
    'wiener_fft',
    'imu',
    'spectral_raw',
    'spectral_wiener',
    'window_stats',
)


class WindowBuffer:
    """
    Maintains sliding window buffers for EMG and IMU data.
    Handles alignment between EMG (500 Hz) and IMU (100 Hz) streams.
    """

    def __init__(
        self,
        window_size: int = 100,
        stride: int = 25,
        sequence_length: int = 8,
        emg_sampling_rate: int = 500,
        imu_sampling_rate: int = 100,
    ):
        self.window_size = window_size
        self.stride = stride
        self.sequence_length = sequence_length
        self.emg_sampling_rate = emg_sampling_rate
        self.imu_sampling_rate = imu_sampling_rate

        # Raw sample buffers
        self.emg_buffer = deque(maxlen=window_size * 2)
        self.imu_buffer: deque = deque(maxlen=int(imu_sampling_rate * 2))

        # Processed-feature sequence buffer (holds normalised window dicts)
        self.window_sequence: deque = deque(maxlen=sequence_length)

        self.window_start_idx = 0

    # ──────────────────────────────────────────────────────────────────────
    # Raw sample ingestion
    # ──────────────────────────────────────────────────────────────────────

    def add_emg_samples(self, emg_samples: np.ndarray):
        self.emg_buffer.extend(emg_samples.flatten())

    def add_imu_samples(
        self,
        imu_samples: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ):
        if imu_samples.ndim == 1:
            imu_samples = imu_samples.reshape(1, -1)

        if not self.imu_buffer:
            base_time = 0.0
        else:
            base_time = self.imu_buffer[-1][0]

        dt_ms = 1000.0 / self.imu_sampling_rate
        for i, sample in enumerate(imu_samples):
            if timestamps is not None and i < len(timestamps):
                ts = timestamps[i]
            else:
                ts = base_time + (i + 1) * dt_ms
            self.imu_buffer.append((ts, sample))

    # ──────────────────────────────────────────────────────────────────────
    # Window creation
    # ──────────────────────────────────────────────────────────────────────

    def can_create_window(self) -> bool:
        return self.window_start_idx + self.window_size <= len(self.emg_buffer)

    def create_window(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract the next EMG window and corresponding IMU window.

        Returns (emg_window, imu_window) or None if not enough data.
        """
        if not self.can_create_window():
            return None

        emg_list = list(self.emg_buffer)
        start = self.window_start_idx
        end = start + self.window_size
        emg_window = np.array(emg_list[start:end], dtype=np.float32)

        # Estimate the time span of this EMG window for IMU alignment
        window_ms = (self.window_size / self.emg_sampling_rate) * 1000.0
        if self.imu_buffer:
            latest_imu_ts = self.imu_buffer[-1][0]
            w_end_ms = latest_imu_ts
            w_start_ms = w_end_ms - window_ms
        else:
            w_start_ms = 0.0
            w_end_ms = window_ms

        imu_window = self.get_imu_window(w_start_ms, w_end_ms)

        # Advance stride, reset if we've run out of buffered data
        self.window_start_idx += self.stride
        buf_len = len(self.emg_buffer)
        if self.window_start_idx + self.window_size > buf_len:
            max_start = buf_len - self.window_size
            if max_start > 0:
                self.window_start_idx = max(
                    0, max_start - (self.window_size - self.stride)
                )

        return emg_window, imu_window

    def get_imu_window(
        self, start_time_ms: float, end_time_ms: float
    ) -> np.ndarray:
        """Return IMU samples in [start_time_ms, end_time_ms] as (n, 6) array."""
        samples = [s for ts, s in self.imu_buffer if start_time_ms <= ts <= end_time_ms]
        if not samples:
            dur_ms = end_time_ms - start_time_ms
            n = max(1, int(dur_ms / (1000.0 / self.imu_sampling_rate)))
            return np.zeros((n, 6), dtype=np.float32)
        return np.array(samples, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Sequence management
    # ──────────────────────────────────────────────────────────────────────

    def can_create_sequence(self) -> bool:
        return len(self.window_sequence) >= self.sequence_length

    def update_sequence(self, window_features: Dict[str, np.ndarray]) -> bool:
        """
        Append a normalised window-feature dict to the sequence buffer.

        Returns True when the sequence buffer is full and ready for inference.
        """
        self.window_sequence.append(window_features)
        return len(self.window_sequence) >= self.sequence_length

    def get_sequence(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Stack the current sequence into arrays ready for EMGLSTMModel.

        Returned shapes (seq_len = sequence_length):
            raw, rms_lms, wiener_td  → (seq_len, window_size)
            wiener_fft               → (seq_len, 64)
            imu                      → (seq_len, 6, n_imu_samples)
            spectral_raw,
            spectral_wiener          → (seq_len, 5)
            window_stats             → (seq_len, 16)
        """
        if not self.can_create_sequence():
            return None

        windows = list(self.window_sequence)

        # Verify all required keys are present in every window
        if not all(k in w for w in windows for k in _MODEL_FEATURE_KEYS):
            return None

        result: Dict[str, np.ndarray] = {}
        for key in _MODEL_FEATURE_KEYS:
            result[key] = np.stack([w[key] for w in windows])
        return result

    def clear(self):
        """Flush all buffers — call on device disconnect / data timeout."""
        self.emg_buffer.clear()
        self.imu_buffer.clear()
        self.window_sequence.clear()
        self.window_start_idx = 0
