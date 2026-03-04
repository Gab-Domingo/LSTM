"""
Sliding window buffer for real-time data processing.
Maintains buffers for EMG and IMU data with proper alignment.
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class WindowBuffer:
    """
    Maintains sliding window buffers for EMG and IMU data.
    Handles data alignment between different sampling rates.
    """
    
    def __init__(self,
                 window_size: int = 100,
                 stride: int = 25,
                 sequence_length: int = 8,
                 emg_sampling_rate: int = 500,
                 imu_sampling_rate: int = 100):
        """
        Args:
            window_size: Number of EMG samples per window
            stride: Stride between windows (samples)
            sequence_length: Number of windows per sequence
            emg_sampling_rate: EMG sampling rate (Hz)
            imu_sampling_rate: IMU sampling rate (Hz)
        """
        self.window_size = window_size
        self.stride = stride
        self.sequence_length = sequence_length
        self.emg_sampling_rate = emg_sampling_rate
        self.imu_sampling_rate = imu_sampling_rate
        
        # Buffer for EMG samples
        self.emg_buffer = deque(maxlen=window_size * 2)  # Keep extra for overlap
        
        # Buffer for IMU samples with timestamps
        self.imu_buffer = deque(maxlen=int(imu_sampling_rate * 2))  # 2 seconds
        
        # Sequence buffer for windows
        self.window_sequence = deque(maxlen=sequence_length)
        
        # Track window position for stride handling
        self.window_start_idx = 0
        
    def add_emg_samples(self, emg_samples: np.ndarray):
        """
        Add EMG samples to buffer.
        
        Args:
            emg_samples: Array of EMG values
        """
        self.emg_buffer.extend(emg_samples.flatten())
    
    def add_imu_samples(self, imu_samples: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """
        Add IMU samples to buffer.
        
        Args:
            imu_samples: Array of IMU values (n_samples, 6) - [ax, ay, az, gx, gy, gz]
            timestamps: Optional timestamps (if None, uses current time)
        """
        if timestamps is None:
            # Generate timestamps assuming uniform sampling
            if len(self.imu_buffer) == 0:
                current_time = 0.0
            else:
                last_ts = self.imu_buffer[-1][0]
                dt = 1.0 / self.imu_sampling_rate * 1000.0  # ms
                current_time = last_ts + dt
        
        if imu_samples.ndim == 1:
            imu_samples = imu_samples.reshape(1, -1)
        
        for i, sample in enumerate(imu_samples):
            if timestamps is not None and i < len(timestamps):
                ts = timestamps[i]
            else:
                ts = current_time + (i / self.imu_sampling_rate * 1000.0)
            
            self.imu_buffer.append((ts, sample))
    
    def get_imu_window(self, start_time_ms: float, end_time_ms: float) -> np.ndarray:
        """
        Get IMU samples within a time window.
        
        Args:
            start_time_ms: Start time (ms)
            end_time_ms: End time (ms)
            
        Returns:
            IMU data array (n_samples, 6)
        """
        imu_samples = []
        for ts, sample in self.imu_buffer:
            if start_time_ms <= ts <= end_time_ms:
                imu_samples.append(sample)
        
        if len(imu_samples) == 0:
            # No IMU data in window, return zeros
            window_duration_ms = end_time_ms - start_time_ms
            expected_samples = int(window_duration_ms / (1000.0 / self.imu_sampling_rate))
            expected_samples = max(1, expected_samples)
            return np.zeros((expected_samples, 6), dtype=np.float32)
        
        return np.array(imu_samples, dtype=np.float32)
    
    def can_create_window(self) -> bool:
        """
        Check if we have enough data to create a window at current position.
        
        Returns:
            True if enough data available
        """
        # Need enough samples from window_start_idx to window_start_idx + window_size
        return self.window_start_idx + self.window_size <= len(self.emg_buffer)
    
    def create_window(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Create a window from buffered data at current position.
        Advances window position by stride after creation.
        
        Returns:
            Tuple of (emg_window, imu_window) or None if not enough data
        """
        if not self.can_create_window():
            return None
        
        # Get EMG window from current position
        emg_list = list(self.emg_buffer)
        start_idx = self.window_start_idx
        end_idx = start_idx + self.window_size
        
        emg_window = np.array(emg_list[start_idx:end_idx], dtype=np.float32)
        
        # Estimate time window for IMU alignment
        # Assuming EMG is sampled at 500Hz, each sample is 2ms
        window_duration_ms = (self.window_size / self.emg_sampling_rate) * 1000.0
        window_start_ms = (start_idx / self.emg_sampling_rate) * 1000.0
        window_end_ms = (end_idx / self.emg_sampling_rate) * 1000.0
        
        # Adjust using IMU timestamps if available
        if len(self.imu_buffer) > 0:
            # Estimate based on buffer position
            latest_imu_time = self.imu_buffer[-1][0]
            buffer_duration_ms = len(self.emg_buffer) / self.emg_sampling_rate * 1000.0
            window_end_ms = latest_imu_time
            window_start_ms = window_end_ms - window_duration_ms
        else:
            # Use relative timing
            window_start_ms = 0.0
            window_end_ms = window_duration_ms
        
        # Get IMU window
        imu_window = self.get_imu_window(window_start_ms, window_end_ms)
        
        # Advance window position by stride
        self.window_start_idx += self.stride
        
        # If we've advanced too far, reset to near the end (keep some overlap)
        buffer_len = len(self.emg_buffer)
        if self.window_start_idx + self.window_size > buffer_len:
            # Reset to allow some buffer room but maintain stride
            max_start = buffer_len - self.window_size
            if max_start > 0:
                self.window_start_idx = max(0, max_start - (self.window_size - self.stride))
        
        return emg_window, imu_window
    
    def can_create_sequence(self) -> bool:
        """
        Check if we can create a sequence of windows.
        
        Returns:
            True if sequence can be created
        """
        return len(self.window_sequence) >= self.sequence_length
    
    def update_sequence(self, window_features: dict) -> bool:
        """
        Update sequence buffer with new window features.
        Returns True if sequence is ready for inference.
        
        Args:
            window_features: Dictionary of window features
            
        Returns:
            True if sequence is ready
        """
        self.window_sequence.append(window_features)
        return len(self.window_sequence) >= self.sequence_length
    
    def get_sequence(self) -> Optional[dict]:
        """
        Get current sequence for inference.
        
        Returns:
            Dictionary of sequences or None if not ready
        """
        if not self.can_create_sequence():
            return None
        
        # Stack windows into sequences
        emg_envelopes = []
        emg_bandpowers = []
        imu_features = []
        vs_rest_features = []
        
        for window in self.window_sequence:
            emg_envelopes.append(window['emg_envelope'])
            emg_bandpowers.append(window['emg_bandpowers'])
            imu_features.append(window['imu_features'])
            # vs_rest is optional (may not be in all windows)
            vs_rest_features.append(window.get('vs_rest', np.array([0.0, 0.0], dtype=np.float32)))
        
        result = {
            'emg_envelope': np.stack(emg_envelopes),
            'emg_bandpowers': np.stack(emg_bandpowers),
            'imu_features': np.stack(imu_features)
        }
        
        # Add vs_rest if available
        if len(vs_rest_features) > 0:
            result['vs_rest'] = np.stack(vs_rest_features)
        
        return result
    
    def clear(self):
        """
        Clear all buffers to remove stale data.
        Useful when ESP32 disconnects or data becomes stale.
        """
        self.emg_buffer.clear()
        self.imu_buffer.clear()
        self.window_sequence.clear()
        self.window_start_idx = 0

