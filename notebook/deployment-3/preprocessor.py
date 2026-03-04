"""
Real-time feature extraction and preprocessing for deployment.
Matches exact algorithms from data_collection/filters.py to ensure compatibility with trained model.

CRITICAL: Uses global StandardScaler normalization (as in training) to maintain feature distribution.
REST baseline is used ONLY for computing vs_rest features, not for normalizing main features.
"""

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import butter, sosfilt
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class RealTimePreprocessor:
    """
    Real-time preprocessing pipeline that EXACTLY matches data collection preprocessing.
    Processing order: Bandpass (20-200Hz) → LMS Adaptive → RMS → Wiener FFT
    
    This ensures features match training data and model expectations.
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 sampling_rate: int = 500,
                 imu_sampling_rate: int = 100,
                 scalers: Optional[Dict] = None,
                 enable_rest_baseline: bool = True):
        """
        Args:
            window_size: Number of EMG samples per window (100 = 200ms @ 500Hz)
            sampling_rate: EMG sampling rate (Hz)
            imu_sampling_rate: IMU sampling rate (Hz)
            scalers: Dictionary of StandardScaler objects for normalization (MUST match training)
            enable_rest_baseline: Whether to collect REST baseline for vs_rest features
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.imu_sampling_rate = imu_sampling_rate
        self.scalers = scalers or {}
        self.enable_rest_baseline = enable_rest_baseline
        
        # ============ Bandpass Filter (20-200 Hz, order 4) ============
        # Matches data_collection/filters.py exactly
        nyq = 0.5 * sampling_rate
        low = 20.0 / nyq
        high = 200.0 / nyq
        if high >= 1.0:
            high = 0.95
        self.bandpass_sos = butter(4, [low, high], btype='band', output='sos')
        self.bandpass_zi = np.zeros((self.bandpass_sos.shape[0], 2))
        
        # ============ LMS Adaptive Filter ============
        # Matches data_collection/filters.py: 16 taps, μ increased for faster adaptation
        self.lms_filter_length = 16
        self.lms_mu = 0.0004  # INCREASED from 0.0002 for faster motion artifact rejection
        self.lms_weights = np.zeros(self.lms_filter_length, dtype=np.float32)
        self.lms_reference_buffer = np.zeros(self.lms_filter_length, dtype=np.float32)
        
        # Normalization statistics (online estimation)
        self.lms_emg_mean = 2048.0  # ADC midpoint
        self.lms_emg_var = 250000.0
        self.lms_gyro_var = 10000.0
        self.lms_alpha = 0.001  # EMA coefficient
        
        # LMS adaptation tracking
        self.lms_adaptation_count = 0
        self.lms_min_adaptation_samples = 100  # Need at least 100 samples before trusting LMS output
        
        # ============ RMS Filter ============
        # Matches data_collection/filters.py: 100 samples window, 50% overlap
        self.rms_window_size = 100
        self.rms_stride = 50  # 50% overlap
        self.rms_buffer = deque(maxlen=self.rms_window_size)
        
        # ============ Wiener Filter ============
        # Matches data_collection/filters.py: 128-point FFT, SNR threshold 2.0
        self.wiener_fft_size = 128  # Will produce 64 bins via rfft
        self.wiener_snr_threshold = 2.0
        self.wiener_noise_samples = 500
        self.wiener_buffer = deque(maxlen=self.wiener_fft_size * 2)
        self.wiener_noise_power = None
        self.wiener_is_calibrated = False
        self.wiener_sample_count = 0
        
        # REST baseline storage
        self.rest_baseline = None
        self.rest_baseline_collected = False
        
        # Diagnostic mode flag (disable after calibration completes)
        self.enable_diagnostics = True
        
        # Signal validity thresholds
        self.signal_validity_threshold = 5.0     # Minimum peak-to-peak for valid signal
        self.rest_activity_threshold = 300.0     # Maximum envelope for REST state (increased - EMG naturally has baseline activity)
        self.active_activity_threshold = 500.0   # Minimum envelope for ACTIVE state (increased for strong contractions)
    
    def check_signal_validity(self, raw_emg: np.ndarray) -> str:
        """
        Check signal validity and state.
        
        Returns:
            'NO_SIGNAL': Sensor detached or invalid (all zeros or constant)
            'REST': Valid signal but quiet (below active threshold)
            'ACTIVE': Valid signal with activity (above active threshold)
        """
        # Check for completely zero or constant signal
        if len(raw_emg) == 0:
            return 'NO_SIGNAL'
        
        raw_range = np.max(raw_emg) - np.min(raw_emg)
        raw_std = np.std(raw_emg)
        raw_mean = np.mean(raw_emg)
        
        # NO_SIGNAL: All zeros or constant (no variation)
        if raw_range < self.signal_validity_threshold or raw_std < 1.0:
            return 'NO_SIGNAL'
        
        # For REST vs ACTIVE, we need envelope - but we can use raw peak-to-peak as proxy
        # This is a preliminary check; final state determined after envelope computation
        return 'REST'  # Will be refined after envelope computation
    
    def check_envelope_state(self, envelope: np.ndarray, bandpowers: np.ndarray) -> str:
        """
        Check state based on envelope (after filtering).
        RELAXED to allow near-zero REST baselines (valid for some sensors).
        
        Returns:
            'NO_SIGNAL': Complete signal loss (no variation at all)
            'REST': Valid signal but quiet (low amplitude is OK)
            'ACTIVE': Significant activity detected
        """
        env_mean = np.mean(envelope)
        env_max = np.max(envelope)
        env_std = np.std(envelope)
        total_bandpower = np.sum(bandpowers) if len(bandpowers) >= 3 else 0.0
        
        # NO_SIGNAL: ONLY if there's absolutely no variation
        # Allow near-zero signals as long as there's SOME variation
        if env_std < 0.001 and total_bandpower < 0.001:
            return 'NO_SIGNAL'
        
        # ACTIVE: Significant activity detected (relative to baseline)
        if env_mean > self.active_activity_threshold or env_std > 20.0:
            return 'ACTIVE'
        
        # REST: Valid signal (can be near-zero, model detects CHANGES)
        return 'REST'
    
    def _bandpass_filter(self, sample: float) -> float:
        """
        Apply bandpass filter (20-200 Hz) to single sample.
        Matches data_collection/filters.py BandpassFilter.process()
        """
        filtered, self.bandpass_zi = sosfilt(self.bandpass_sos, [sample], zi=self.bandpass_zi)
        return filtered[0]
    
    def _lms_filter_sample(self, emg_sample: float, gyro_mag: float) -> float:
        """
        Apply corrected LMS adaptive filtering.
        
        MODIFIED FOR DEPLOYMENT: Gyroscope input is DISABLED (gyro_mag forced to 0).
        This prevents motion from being filtered out as artifacts, allowing wheelchair movement.
        The filter effectively acts as a passthrough after bandpass filtering.
        
        Args:
            emg_sample: Bandpass-filtered EMG sample
            gyro_mag: Gyroscope magnitude (IGNORED - disabled for deployment)
            
        Returns:
            filtered_emg: EMG sample (LMS disabled, only bandpass applied)
        """
        # DEPLOYMENT MODIFICATION: Force gyro to zero to disable motion artifact removal
        gyro_mag = 0.0
        
        # Update statistics and center EMG (CORRECT order)
        emg_centered = emg_sample - self.lms_emg_mean
        
        # Update mean AFTER centering
        self.lms_emg_mean = (1 - self.lms_alpha) * self.lms_emg_mean + self.lms_alpha * emg_sample
        
        # Update variance
        self.lms_emg_var = (1 - self.lms_alpha) * self.lms_emg_var + self.lms_alpha * (emg_centered ** 2)
        self.lms_gyro_var = (1 - self.lms_alpha) * self.lms_gyro_var + self.lms_alpha * (gyro_mag ** 2)
        
        # Normalize
        emg_std = np.sqrt(self.lms_emg_var + 1e-8)
        gyro_std = np.sqrt(self.lms_gyro_var + 1e-8)
        emg_norm = emg_centered / emg_std
        gyro_norm = gyro_mag / gyro_std
        
        # Update reference buffer (FIFO)
        self.lms_reference_buffer = np.roll(self.lms_reference_buffer, 1)
        self.lms_reference_buffer[0] = gyro_norm
        
        # Estimate motion artifact (will be ~0 since gyro_norm = 0)
        estimated_artifact = np.dot(self.lms_weights, self.lms_reference_buffer)
        
        # Compute error (cleaned signal)
        error = emg_norm - estimated_artifact
        
        # Normalized LMS weight update (effectively disabled since gyro = 0)
        reference_power = np.dot(self.lms_reference_buffer, self.lms_reference_buffer)
        if reference_power > 1e-6:
            normalized_mu = self.lms_mu / (reference_power + 0.1)
            weight_delta = normalized_mu * error * self.lms_reference_buffer
            weight_delta = np.clip(weight_delta, -0.02, 0.02)
            self.lms_weights += weight_delta
            self.lms_weights = np.clip(self.lms_weights, -5.0, 5.0)
        
        # Denormalize
        filtered_emg_centered = error * emg_std
        filtered_emg = filtered_emg_centered + self.lms_emg_mean
        
        return filtered_emg
    
    def _compute_rms_on_lms(self, lms_signal: np.ndarray) -> np.ndarray:
        """
        Compute RMS envelope from LMS-filtered signal.
        Matches data_collection/filters.py RMSFilter but returns full-rate envelope.
        
        For real-time: Compute RMS on LMS signal using sliding window.
        Returns full-rate RMS envelope (not downsampled like in data collection).
        """
        if len(lms_signal) < self.rms_window_size:
            # If signal is shorter than window, compute single RMS value
            rms_value = np.sqrt(np.mean(lms_signal ** 2))
            return np.full(len(lms_signal), rms_value)
        
        rms_envelope = np.zeros_like(lms_signal, dtype=np.float32)
        half_window = self.rms_window_size // 2
        
        for i in range(len(lms_signal)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(lms_signal), i + half_window + 1)
            window = lms_signal[start_idx:end_idx]
            rms_envelope[i] = np.sqrt(np.mean(window ** 2))
        
        return rms_envelope
    
    def _wiener_filter_fft(self, signal_window: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filter in frequency domain.
        Matches data_collection/filters.py WienerFilter._wiener_filter_fft() EXACTLY.
        
        Args:
            signal_window: Signal window (wiener_fft_size samples)
            
        Returns:
            fft_magnitudes: FFT magnitude bins (64 bins from 128-point rfft)
        """
        # Apply FFT (real FFT for real signal)
        fft_signal = rfft(signal_window, n=self.wiener_fft_size)
        signal_power = np.abs(fft_signal) ** 2
        
        if not self.wiener_is_calibrated:
            # During calibration, return unfiltered magnitudes
            return np.abs(fft_signal).astype(np.float32)
        
        # Calculate Wiener gain: H(f) = max(0, 1 - noise_power/signal_power)
        wiener_gain = np.maximum(0, 1 - (self.wiener_noise_power / (signal_power + 1e-10)))
        
        # Apply SNR-based thresholding - REDUCED threshold to preserve more frequency content
        # The model needs frequency information to distinguish REST from FIST
        snr = signal_power / (self.wiener_noise_power + 1e-10)
        
        # Use a LOWER threshold (1.0 instead of 2.0) to preserve more frequency content
        # This helps the model see the frequency differences between REST and FIST
        wiener_threshold = 1.0  # REDUCED from 2.0 to preserve more signal
        
        # Apply threshold - keep frequencies with SNR > 1.0
        wiener_gain = np.where(snr > wiener_threshold, wiener_gain, 0.0)
        
        # Apply filter in frequency domain
        filtered_fft = fft_signal * wiener_gain
        
        # Return magnitude spectrum (64 bins from 128-point rfft)
        return np.abs(filtered_fft).astype(np.float32)
    
    def compute_coarse_bandpowers(self, fft_magnitudes: np.ndarray) -> np.ndarray:
        """
        Compute 3 coarse bandpowers from Wiener FFT magnitudes.
        Matches training bandpower computation EXACTLY.
        
        Args:
            fft_magnitudes: FFT magnitude bins (64,)
            
        Returns:
            Bandpowers [low, mid, high] (3,)
        """
        n_fft = len(fft_magnitudes)
        freqs = np.linspace(0, self.sampling_rate/2, n_fft)
        
        # Define frequency bands (matching training)
        low_mask = (freqs >= 20) & (freqs <= 60)
        mid_mask = (freqs >= 60) & (freqs <= 150)
        high_mask = (freqs >= 150) & (freqs <= 250)
        
        # Compute bandpowers (sum of squared magnitudes)
        bandpower_low = np.sum(fft_magnitudes[low_mask] ** 2) if np.any(low_mask) else 0.0
        bandpower_mid = np.sum(fft_magnitudes[mid_mask] ** 2) if np.any(mid_mask) else 0.0
        bandpower_high = np.sum(fft_magnitudes[high_mask] ** 2) if np.any(high_mask) else 0.0
        
        return np.array([bandpower_low, bandpower_mid, bandpower_high], dtype=np.float32)
    
    def extract_imu_stats(self, imu_window: np.ndarray) -> np.ndarray:
        """
        Extract IMU magnitude statistics from window.
        Matches training feature extraction EXACTLY.
        
        Args:
            imu_window: IMU data (n_samples, 6) - [ax, ay, az, gx, gy, gz]
            
        Returns:
            IMU features [gyro_mag_mean, accel_mag_mean, gyro_mag_std, accel_mag_std] (4,)
        """
        if imu_window.ndim == 1:
            imu_window = imu_window.reshape(1, -1)
        
        if imu_window.shape[1] != 6:
            raise ValueError(f"Expected IMU data with 6 channels, got {imu_window.shape[1]}")
        
        accel = imu_window[:, :3]  # (n, 3)
        gyro = imu_window[:, 3:]   # (n, 3)
        
        # Compute magnitudes
        accel_mag = np.linalg.norm(accel, axis=1)  # (n,)
        gyro_mag = np.linalg.norm(gyro, axis=1)    # (n,)
        
        return np.array([
            np.mean(gyro_mag),
            np.mean(accel_mag),
            np.std(gyro_mag),
            np.std(accel_mag)
        ], dtype=np.float32)
    
    def process_window(self, 
                      raw_emg: np.ndarray,
                      imu_data: np.ndarray,
                      device_id: int = None) -> Dict[str, np.ndarray]:
        """
        Process a single window of data through the complete pipeline.
        Matches data_collection/filters.py FilterPipeline processing order EXACTLY.
        
        Processing order:
        1. Bandpass filter (20-200 Hz)
        2. LMS adaptive filter (motion artifact removal)
        3. RMS envelope (computed on LMS-filtered signal)
        4. Wiener FFT (for bandpowers)
        5. Combine RMS+LMS into envelope
        
        Args:
            raw_emg: Raw EMG signal window (window_size,)
            imu_data: IMU data window (n_imu_samples, 6)
            device_id: Device ID for diagnostics (1=LEFT, 2=RIGHT)
            
        Returns:
            Dictionary with extracted features:
                - 'emg_envelope': RMS+LMS combined envelope (window_size,)
                - 'emg_bandpowers': 3 bandpowers [low, mid, high]
                - 'imu_features': 4 IMU stats
        """
        # Initialize diagnostic counter if needed
        if device_id == 2:
            if not hasattr(self, '_diagnostic_counter'):
                self._diagnostic_counter = 0
            self._diagnostic_counter += 1
        
        # Ensure correct window size
        if len(raw_emg) != self.window_size:
            if len(raw_emg) < self.window_size:
                raw_emg = np.interp(
                    np.linspace(0, len(raw_emg)-1, self.window_size),
                    np.arange(len(raw_emg)),
                    raw_emg
                )
            else:
                raw_emg = raw_emg[:self.window_size]
        
        # DIAGNOSTIC: Track Device 2 signal through pipeline (only if enabled)
        if self.enable_diagnostics and device_id == 2 and self._diagnostic_counter % 20 == 0:
            raw_mean = np.mean(raw_emg)
            raw_std = np.std(raw_emg)
            raw_range = (np.min(raw_emg), np.max(raw_emg))
            print(f"\n[Device 2 Pipeline] Window #{self._diagnostic_counter}")
            print(f"  Step 1 - Raw EMG: mean={raw_mean:.2f}, std={raw_std:.2f}, range=[{raw_range[0]:.0f}, {raw_range[1]:.0f}]")
        
        # Extract gyroscope magnitude for LMS motion cancellation
        if imu_data.ndim == 1:
            imu_data = imu_data.reshape(1, -1)
        
        if imu_data.shape[1] >= 6:
            gyro = imu_data[:, 3:6]  # Extract gx, gy, gz
            gyro_mag = np.linalg.norm(gyro, axis=1)  # sqrt(gx² + gy² + gz²)
        else:
            gyro_mag = np.zeros(len(raw_emg))
        
        # Interpolate gyro_mag to match EMG sampling rate if needed
        if len(gyro_mag) != len(raw_emg):
            if len(gyro_mag) > 1:
                gyro_mag = np.interp(
                    np.linspace(0, len(gyro_mag)-1, len(raw_emg)),
                    np.arange(len(gyro_mag)),
                    gyro_mag
                )
            else:
                gyro_mag = np.full(len(raw_emg), gyro_mag[0] if len(gyro_mag) > 0 else 0.0)
        
        # ============ STEP 1: Bandpass Filter (20-200 Hz) ============
        bandpass_filtered = np.zeros_like(raw_emg, dtype=np.float32)
        for i in range(len(raw_emg)):
            bandpass_filtered[i] = self._bandpass_filter(raw_emg[i])
        
        # DIAGNOSTIC: Check bandpass output
        if self.enable_diagnostics and device_id == 2 and self._diagnostic_counter % 20 == 0:
            bp_mean = np.mean(bandpass_filtered)
            bp_std = np.std(bandpass_filtered)
            bp_range = (np.min(bandpass_filtered), np.max(bandpass_filtered))
            print(f"  Step 2 - Bandpass: mean={bp_mean:.2f}, std={bp_std:.2f}, range=[{bp_range[0]:.2f}, {bp_range[1]:.2f}]")
        
        # ============ STEP 2: LMS Adaptive Filter ============
        # Compute average gyroscope magnitude for this window (motion indicator)
        avg_gyro_mag = np.mean(gyro_mag) if len(gyro_mag) > 0 else 0.0
        
        lms_filtered = np.zeros_like(bandpass_filtered, dtype=np.float32)
        for i in range(len(bandpass_filtered)):
            lms_value = self._lms_filter_sample(bandpass_filtered[i], gyro_mag[i])
            self.lms_adaptation_count += 1
            
            # Validation check (matches data collection)
            if not np.isfinite(lms_value):
                # Reset LMS filter on divergence
                self.lms_weights = np.zeros(self.lms_filter_length, dtype=np.float32)
                self.lms_reference_buffer = np.zeros(self.lms_filter_length, dtype=np.float32)
                self.lms_adaptation_count = 0  # Reset adaptation counter
                lms_value = bandpass_filtered[i]
            
            # If LMS hasn't adapted enough yet, use bandpass-filtered signal
            # This prevents motion artifacts from passing through during initial adaptation
            if self.lms_adaptation_count < self.lms_min_adaptation_samples:
                # Use bandpass only until LMS has adapted
                # This is conservative but prevents false FIST predictions from motion
                lms_filtered[i] = bandpass_filtered[i]
            else:
                lms_filtered[i] = lms_value
        
        # DIAGNOSTIC: Check LMS output
        if self.enable_diagnostics and device_id == 2 and self._diagnostic_counter % 20 == 0:
            lms_mean = np.mean(lms_filtered)
            lms_std = np.std(lms_filtered)
            lms_range = (np.min(lms_filtered), np.max(lms_filtered))
            print(f"  Step 3 - LMS: mean={lms_mean:.2f}, std={lms_std:.2f}, range=[{lms_range[0]:.2f}, {lms_range[1]:.2f}]")
            print(f"           LMS adapted: {self.lms_adaptation_count >= self.lms_min_adaptation_samples}, count={self.lms_adaptation_count}")
        
        # ============ STEP 3: RMS Envelope (computed on LMS-filtered signal) ============
        rms_envelope = self._compute_rms_on_lms(lms_filtered)
        
        # DIAGNOSTIC: Check RMS envelope
        if self.enable_diagnostics and device_id == 2 and self._diagnostic_counter % 20 == 0:
            rms_mean = np.mean(rms_envelope)
            rms_std = np.std(rms_envelope)
            rms_range = (np.min(rms_envelope), np.max(rms_envelope))
            print(f"  Step 4 - RMS: mean={rms_mean:.2f}, std={rms_std:.2f}, range=[{rms_range[0]:.2f}, {rms_range[1]:.2f}]")
        
        # ============ STEP 4: Wiener FFT (for bandpowers) ============
        # Add samples to Wiener buffer for calibration
        for sample in bandpass_filtered:
            self.wiener_buffer.append(sample)
            self.wiener_sample_count += 1
        
        # Calibrate noise model if needed
        if not self.wiener_is_calibrated and self.wiener_sample_count >= self.wiener_noise_samples:
            noise_data = np.array(list(self.wiener_buffer)[-self.wiener_noise_samples:])
            fft_noise = rfft(noise_data, n=self.wiener_fft_size)
            self.wiener_noise_power = np.abs(fft_noise) ** 2
            
            # CRITICAL FIX: For near-zero signals, set minimum noise floor
            # This prevents Wiener filter from zeroing out everything
            min_noise_floor = 0.01  # Minimum noise power per frequency bin
            self.wiener_noise_power = np.maximum(self.wiener_noise_power, min_noise_floor)
            
            self.wiener_is_calibrated = True
        
        # Compute Wiener FFT on current window
        if len(self.wiener_buffer) >= self.wiener_fft_size:
            # Use last fft_size samples
            wiener_window = np.array(list(self.wiener_buffer)[-self.wiener_fft_size:])
        else:
            # Pad with zeros if buffer not full
            wiener_window = np.zeros(self.wiener_fft_size, dtype=np.float32)
            wiener_window[-len(self.wiener_buffer):] = list(self.wiener_buffer)
        
        wiener_fft_mags = self._wiener_filter_fft(wiener_window)
        emg_bandpowers = self.compute_coarse_bandpowers(wiener_fft_mags)
        
        # DIAGNOSTIC: Check Wiener and bandpowers
        if self.enable_diagnostics and device_id == 2 and self._diagnostic_counter % 20 == 0:
            wiener_mean = np.mean(wiener_fft_mags)
            wiener_max = np.max(wiener_fft_mags)
            wiener_nonzero = np.sum(wiener_fft_mags > 0.1)
            print(f"  Step 5 - Wiener FFT: mean={wiener_mean:.2f}, max={wiener_max:.2f}, nonzero_bins={wiener_nonzero}/64, calibrated={self.wiener_is_calibrated}")
            print(f"           Bandpowers: low={emg_bandpowers[0]:.2f}, mid={emg_bandpowers[1]:.2f}, high={emg_bandpowers[2]:.2f}")
        
        # ============ STEP 5: Combine RMS+LMS into single envelope (matches training EXACTLY) ============
        # Training uses: emg_envelope = (rms_window + lms_window) / 2.0
        # This preserves amplitude differences between REST and FIST
        # RMS envelope is computed on LMS-filtered signal (motion artifacts already removed)
        # Combining both provides better amplitude representation
        emg_envelope = (rms_envelope + lms_filtered) / 2.0
        
        # DIAGNOSTIC: Check final envelope
        if self.enable_diagnostics and device_id == 2 and self._diagnostic_counter % 20 == 0:
            env_mean = np.mean(emg_envelope)
            env_std = np.std(emg_envelope)
            env_range = (np.min(emg_envelope), np.max(emg_envelope))
            print(f"  Step 6 - Final Envelope: mean={env_mean:.2f}, std={env_std:.2f}, range=[{env_range[0]:.2f}, {env_range[1]:.2f}]")
            
            # Check for signal loss
            if env_mean < 1.0:
                print(f"  ⚠️  WARNING: Envelope near zero - signal being destroyed!")
            if emg_bandpowers[2] < 1.0:
                print(f"  ⚠️  WARNING: High bandpower near zero - Wiener filter too aggressive!")
            print()
        
        # ============ STEP 6: Extract IMU statistics ============
        # DEVICE-SPECIFIC: Disable IMU for Device 1 (LEFT arm) to avoid motion artifacts
        # Device 1 = LEFT arm: pure EMG only (no IMU interference)
        # Device 2 = RIGHT arm: can use IMU if needed
        if device_id == 1:
            # Zero out IMU features for Device 1 (LEFT arm)
            imu_features = np.zeros(4, dtype=np.float32)
        else:
            # Device 2 (RIGHT arm) or unknown device: use normal IMU features
            imu_features = self.extract_imu_stats(imu_data)
        
        return {
            'emg_envelope': emg_envelope.astype(np.float32),
            'emg_bandpowers': emg_bandpowers,
            'imu_features': imu_features
        }
    
    def collect_rest_baseline(self, features: Dict[str, np.ndarray]):
        """
        Collect REST baseline statistics from a window.
        Should be called during session initialization (2 seconds of REST data).
        
        Args:
            features: Dictionary of feature arrays from REST windows
        """
        if self.rest_baseline is None:
            self.rest_baseline = {
                'emg_envelope': {'values': [], 'mean': None, 'std': None},
                'emg_bandpowers': {'values': [], 'mean': None, 'std': None},
                'imu_features': {'values': [], 'mean': None, 'std': None}
            }
        
        # Collect feature values
        self.rest_baseline['emg_envelope']['values'].append(features['emg_envelope'].flatten())
        self.rest_baseline['emg_bandpowers']['values'].append(features['emg_bandpowers'])
        self.rest_baseline['imu_features']['values'].append(features['imu_features'])
    
    def finalize_rest_baseline(self):
        """
        Finalize REST baseline by computing statistics from collected samples.
        Should be called after collecting enough REST windows (e.g., 2 seconds).
        
        ENHANCED: Now validates baseline quality and uses MEDIAN instead of MEAN to reject outliers.
        Noisy baselines will be detected and the user will be warned to recollect.
        """
        if self.rest_baseline is None or len(self.rest_baseline['emg_envelope']['values']) == 0:
            print("⚠️  REST baseline WARNING: no samples collected - using fallback")
            # Use fallback: set minimal baseline to prevent division by zero
            self.rest_baseline = {
                'emg_envelope': {'values': [], 'mean': 5.0, 'std': 3.0},  # Lower fallback for better sensitivity
                'emg_bandpowers': {'values': [], 'mean': np.array([5.0, 10.0, 5.0]), 'std': np.array([2.0, 3.0, 2.0])},
                'imu_features': {'values': [], 'mean': np.array([0.0, 0.0, 0.0, 0.0]), 'std': np.array([1.0, 1.0, 1.0, 1.0])}
            }
            self.rest_baseline_collected = True
            return True
        
        # Compute statistics for each feature type
        # EMG envelope - use MEDIAN instead of MEAN to reject noise outliers
        envelope_all = np.concatenate(self.rest_baseline['emg_envelope']['values'])
        
        # Use median and MAD (median absolute deviation) for robust statistics
        envelope_median = np.median(envelope_all)
        envelope_mad = np.median(np.abs(envelope_all - envelope_median))
        envelope_robust_std = 1.4826 * envelope_mad  # MAD to std conversion
        
        # Use median as mean (more robust to outliers)
        self.rest_baseline['emg_envelope']['mean'] = envelope_median
        self.rest_baseline['emg_envelope']['std'] = max(envelope_robust_std, 1.0) + 1e-10  # Minimum std of 1.0
        
        # Bandpowers - also use median for robustness
        bandpowers_all = np.stack(self.rest_baseline['emg_bandpowers']['values'])
        bp_medians = np.median(bandpowers_all, axis=0)
        bp_mads = np.median(np.abs(bandpowers_all - bp_medians), axis=0)
        bp_robust_stds = 1.4826 * bp_mads
        
        self.rest_baseline['emg_bandpowers']['mean'] = bp_medians
        self.rest_baseline['emg_bandpowers']['std'] = np.maximum(bp_robust_stds, 1.0) + 1e-10  # Minimum std of 1.0
        
        # IMU features
        imu_all = np.stack(self.rest_baseline['imu_features']['values'])
        self.rest_baseline['imu_features']['mean'] = np.mean(imu_all, axis=0)
        self.rest_baseline['imu_features']['std'] = np.std(imu_all, axis=0) + 1e-10
        
        # WARNINGS: Check baseline quality but NEVER reject
        envelope_mean = self.rest_baseline['emg_envelope']['mean']
        envelope_std = self.rest_baseline['emg_envelope']['std']
        bp_high_mean = self.rest_baseline['emg_bandpowers']['mean'][2]
        bp_low_mean = self.rest_baseline['emg_bandpowers']['mean'][0]
        bp_mid_mean = self.rest_baseline['emg_bandpowers']['mean'][1]
        total_bandpower = bp_low_mean + bp_mid_mean + bp_high_mean
        
        # Warning 1: Complete signal loss (sensor disconnected)
        if envelope_mean < 0.01 and envelope_std < 0.01:
            print(f"⚠️  REST baseline WARNING: no signal variation (mean={envelope_mean:.4f}, std={envelope_std:.4f})")
            print(f"    Sensor may be disconnected - check hardware")
            print(f"    Proceeding anyway (may have reduced accuracy)")
        
        # Warning 2: Baseline too high (user may have been contracting)
        elif envelope_mean > self.rest_activity_threshold:
            print(f"⚠️  REST baseline WARNING: envelope high ({envelope_mean:.2f}) - user may have been contracting")
            print(f"    Proceeding anyway (baseline may be suboptimal)")
        
        # Warning 3: High variability (filters may not have stabilized)
        elif envelope_std > envelope_mean * 0.5:
            cv = envelope_std / (envelope_mean + 1e-6)
            print(f"⚠️  REST baseline WARNING: high variation (std={envelope_std:.2f}, mean={envelope_mean:.2f}, CV={cv:.3f})")
            print(f"    Filters may not have stabilized - consider longer warmup")
            print(f"    This may cause poor FIST detection - system will adapt during session")
            print(f"    Proceeding anyway (baseline may be suboptimal)")
        
        # Warning 4: Very low bandpower (signal may be weak)
        elif total_bandpower < 0.01:
            print(f"⚠️  REST baseline WARNING: all bandpowers very low (total={total_bandpower:.4f})")
            print(f"    Signal may be weak or filters too aggressive")
            print(f"    Proceeding anyway (may have reduced accuracy)")
        
        # Baseline is ALWAYS accepted (user is at rest at start)
        self.rest_baseline_collected = True
        
        # Clear collected values to save memory
        self.rest_baseline['emg_envelope']['values'] = []
        self.rest_baseline['emg_bandpowers']['values'] = []
        self.rest_baseline['imu_features']['values'] = []
        
        return True  # ALWAYS return True - never reject baseline
    
    def normalize_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply normalization to features using ONLY global StandardScaler.
        
        CRITICAL: This MUST match the normalization used during training.
        REST baseline is used ONLY for computing vs_rest features, not for normalizing main features.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Normalized features dictionary (using global StandardScaler)
        """
        normalized = {}
        
        # ALWAYS use global StandardScaler normalization (matches training)
        # Normalize EMG envelope
        if 'emg_envelope' in self.scalers:
            envelope_flat = features['emg_envelope'].reshape(-1, 1)
            envelope_norm = self.scalers['emg_envelope'].transform(envelope_flat).flatten()
            normalized['emg_envelope'] = envelope_norm.astype(np.float32)
        else:
            normalized['emg_envelope'] = features['emg_envelope'].astype(np.float32)
        
        # Normalize bandpowers
        if 'emg_bandpowers' in self.scalers:
            bandpowers_reshaped = features['emg_bandpowers'].reshape(1, -1)
            bandpowers_norm = self.scalers['emg_bandpowers'].transform(bandpowers_reshaped).flatten()
            normalized['emg_bandpowers'] = bandpowers_norm.astype(np.float32)
        else:
            normalized['emg_bandpowers'] = features['emg_bandpowers'].astype(np.float32)
        
        # Normalize IMU features
        if 'imu_features' in self.scalers:
            imu_reshaped = features['imu_features'].reshape(1, -1)
            imu_norm = self.scalers['imu_features'].transform(imu_reshaped).flatten()
            normalized['imu_features'] = imu_norm.astype(np.float32)
        else:
            normalized['imu_features'] = features['imu_features'].astype(np.float32)
        
        return normalized
