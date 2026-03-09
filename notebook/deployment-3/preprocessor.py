"""
Real-time feature extraction and preprocessing for deployment.

Matches the LSTM.ipynb training pipeline exactly:
  - Two-step normalisation:
      STEP 1: per-window mean subtraction (mean_only)
      STEP 2: global StandardScaler (loaded from standard_scalers.pkl)
  - Feature set per window:
      raw          (window_size,)   – bandpass-filtered raw EMG
      rms_lms      (window_size,)   – RMS envelope on LMS signal (single ch)
      wiener_td    (window_size,)   – Wiener-filtered time domain
      wiener_fft   (64,)            – Wiener FFT magnitude bins
      imu          (6, n_imu)       – per-time-step z-scored IMU channels
      spectral_raw    (5,)          – [bp_low, bp_mid, bp_high, entropy, centroid] from raw
      spectral_wiener (5,)          – same from wiener_td
      window_stats    (16,)         – 8 time-domain stats from raw + 8 from wiener_td
"""

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import butter, sosfilt
from typing import Dict, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class RealTimePreprocessor:
    """
    Real-time preprocessing pipeline producing features for the multi-branch
    EMGLSTMModel.  Processing order matches data_collection/filters.py exactly:
      Bandpass (20-200 Hz) → LMS (disabled for deployment) → RMS → Wiener FFT
    """

    def __init__(
        self,
        window_size: int = 100,
        sampling_rate: int = 500,
        imu_sampling_rate: int = 100,
        scalers: Optional[Dict] = None,
        enable_rest_baseline: bool = True,
    ):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.imu_sampling_rate = imu_sampling_rate
        self.scalers = scalers or {}
        self.enable_rest_baseline = enable_rest_baseline

        # ── Bandpass filter (20-200 Hz, order 4) ──────────────────────────
        nyq = 0.5 * sampling_rate
        low = 20.0 / nyq
        high = min(200.0 / nyq, 0.95)
        self.bandpass_sos = butter(4, [low, high], btype='band', output='sos')
        self.bandpass_zi = np.zeros((self.bandpass_sos.shape[0], 2))

        # ── LMS adaptive filter ────────────────────────────────────────────
        self.lms_filter_length = 16
        self.lms_mu = 0.0004
        self.lms_weights = np.zeros(self.lms_filter_length, dtype=np.float32)
        self.lms_reference_buffer = np.zeros(self.lms_filter_length, dtype=np.float32)
        self.lms_emg_mean = 2048.0
        self.lms_emg_var = 250000.0
        self.lms_gyro_var = 10000.0
        self.lms_alpha = 0.001
        self.lms_adaptation_count = 0
        self.lms_min_adaptation_samples = 100

        # ── RMS envelope ──────────────────────────────────────────────────
        self.rms_window_size = 100
        self.rms_buffer = deque(maxlen=self.rms_window_size)

        # ── Wiener filter ─────────────────────────────────────────────────
        self.wiener_fft_size = 128          # 128-point FFT → 65 complex bins → 64 usable
        self.wiener_noise_samples = 500
        self.wiener_buffer = deque(maxlen=self.wiener_fft_size * 2)
        self.wiener_noise_power = None
        self.wiener_is_calibrated = False
        self.wiener_sample_count = 0

        # ── REST baseline ─────────────────────────────────────────────────
        self.rest_baseline = None
        self.rest_baseline_collected = False

        # ── Signal diagnostics ─────────────────────────────────────────────
        self.enable_diagnostics = True
        self._diagnostic_counter = 0
        self.signal_validity_threshold = 5.0
        self.rest_activity_threshold = 300.0
        self.active_activity_threshold = 500.0

    # ──────────────────────────────────────────────────────────────────────
    # Signal state helpers (unchanged from original)
    # ──────────────────────────────────────────────────────────────────────

    def check_signal_validity(self, raw_emg: np.ndarray) -> str:
        if len(raw_emg) == 0:
            return 'NO_SIGNAL'
        if (np.max(raw_emg) - np.min(raw_emg)) < self.signal_validity_threshold or np.std(raw_emg) < 1.0:
            return 'NO_SIGNAL'
        return 'REST'

    def check_envelope_state(self, envelope: np.ndarray, bandpowers: np.ndarray) -> str:
        env_std = np.std(envelope)
        total_bp = np.sum(bandpowers) if len(bandpowers) >= 3 else 0.0
        if env_std < 0.001 and total_bp < 0.001:
            return 'NO_SIGNAL'
        if np.mean(envelope) > self.active_activity_threshold or env_std > 20.0:
            return 'ACTIVE'
        return 'REST'

    # ──────────────────────────────────────────────────────────────────────
    # Low-level filter steps
    # ──────────────────────────────────────────────────────────────────────

    def _bandpass_filter(self, sample: float) -> float:
        filtered, self.bandpass_zi = sosfilt(self.bandpass_sos, [sample], zi=self.bandpass_zi)
        return filtered[0]

    def _lms_filter_sample(self, emg_sample: float, gyro_mag: float) -> float:
        """LMS with gyro disabled (forced to 0) for wheelchair deployment."""
        gyro_mag = 0.0
        emg_centered = emg_sample - self.lms_emg_mean
        self.lms_emg_mean = (1 - self.lms_alpha) * self.lms_emg_mean + self.lms_alpha * emg_sample
        self.lms_emg_var = (1 - self.lms_alpha) * self.lms_emg_var + self.lms_alpha * emg_centered ** 2
        self.lms_gyro_var = (1 - self.lms_alpha) * self.lms_gyro_var + self.lms_alpha * gyro_mag ** 2

        emg_std = np.sqrt(self.lms_emg_var + 1e-8)
        gyro_std = np.sqrt(self.lms_gyro_var + 1e-8)
        emg_norm = emg_centered / emg_std
        gyro_norm = gyro_mag / gyro_std

        self.lms_reference_buffer = np.roll(self.lms_reference_buffer, 1)
        self.lms_reference_buffer[0] = gyro_norm

        estimated_artifact = np.dot(self.lms_weights, self.lms_reference_buffer)
        error = emg_norm - estimated_artifact

        ref_power = np.dot(self.lms_reference_buffer, self.lms_reference_buffer)
        if ref_power > 1e-6:
            norm_mu = self.lms_mu / (ref_power + 0.1)
            delta = np.clip(norm_mu * error * self.lms_reference_buffer, -0.02, 0.02)
            self.lms_weights = np.clip(self.lms_weights + delta, -5.0, 5.0)

        return error * emg_std + self.lms_emg_mean

    def _compute_rms_envelope(self, lms_signal: np.ndarray) -> np.ndarray:
        """Sliding-window RMS envelope — full-rate output."""
        if len(lms_signal) < self.rms_window_size:
            return np.full(len(lms_signal), np.sqrt(np.mean(lms_signal ** 2)), dtype=np.float32)
        half = self.rms_window_size // 2
        out = np.zeros(len(lms_signal), dtype=np.float32)
        for i in range(len(lms_signal)):
            w = lms_signal[max(0, i - half): min(len(lms_signal), i + half + 1)]
            out[i] = np.sqrt(np.mean(w ** 2))
        return out

    def _wiener_filter(self, signal_window: np.ndarray):
        """
        Apply Wiener filter in frequency domain.

        Returns
        -------
        wiener_td  : np.ndarray  (window_size,)  – time-domain filtered signal
        fft_mags   : np.ndarray  (64,)           – magnitude spectrum (64 bins)
        """
        fft_signal = rfft(signal_window, n=self.wiener_fft_size)
        signal_power = np.abs(fft_signal) ** 2

        if not self.wiener_is_calibrated:
            # Before calibration: pass-through
            td = np.real(irfft(fft_signal, n=self.wiener_fft_size))[: self.window_size]
            mags = np.abs(fft_signal)[: 64].astype(np.float32)
            return td.astype(np.float32), mags

        wiener_gain = np.maximum(0.0, 1.0 - self.wiener_noise_power / (signal_power + 1e-10))
        snr = signal_power / (self.wiener_noise_power + 1e-10)
        wiener_gain = np.where(snr > 1.0, wiener_gain, 0.0)

        filtered_fft = fft_signal * wiener_gain
        td = np.real(irfft(filtered_fft, n=self.wiener_fft_size))[: self.window_size]
        mags = np.abs(filtered_fft)[: 64].astype(np.float32)
        return td.astype(np.float32), mags

    # ──────────────────────────────────────────────────────────────────────
    # Feature helpers (match dataset._compute_* in LSTM.ipynb)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_spectral_features(self, signal_window: np.ndarray) -> np.ndarray:
        """
        Compute 5 spectral features matching LSTM.ipynb _compute_spectral_features().

        Returns [bandpower_low, bandpower_mid, bandpower_high,
                 spectral_entropy, spectral_centroid]
        """
        n = len(signal_window)
        fft_vals = np.fft.fft(signal_window)
        fft_mag = np.abs(fft_vals[: n // 2])
        freqs = np.fft.fftfreq(n, 1.0 / self.sampling_rate)[: n // 2]

        low_mask = (freqs >= 20) & (freqs <= 60)
        mid_mask = (freqs >= 60) & (freqs <= 150)
        high_mask = (freqs >= 150) & (freqs <= 250)

        bp_low = float(np.sum(fft_mag[low_mask] ** 2)) if np.any(low_mask) else 0.0
        bp_mid = float(np.sum(fft_mag[mid_mask] ** 2)) if np.any(mid_mask) else 0.0
        bp_high = float(np.sum(fft_mag[high_mask] ** 2)) if np.any(high_mask) else 0.0

        power = fft_mag ** 2
        power_norm = power / (np.sum(power) + 1e-10)
        entropy = float(-np.sum(power_norm * np.log(power_norm + 1e-10)))
        centroid = float(np.sum(freqs * power) / (np.sum(power) + 1e-10))

        return np.array([bp_low, bp_mid, bp_high, entropy, centroid], dtype=np.float32)

    def _compute_time_domain_stats(self, signal_window: np.ndarray) -> np.ndarray:
        """
        Compute 8 amplitude-sensitive time-domain stats matching
        LSTM.ipynb _compute_time_domain_stats().

        Returns [mean_abs, rms, peak_to_peak, std,
                 waveform_length, zero_crossing_rate,
                 slope_sign_change_rate, area]
        """
        if signal_window.size == 0:
            return np.zeros(8, dtype=np.float32)

        abs_sig = np.abs(signal_window)
        n = len(signal_window)
        return np.array([
            np.mean(abs_sig),
            np.sqrt(np.mean(signal_window ** 2)),
            np.ptp(signal_window),
            np.std(signal_window),
            np.sum(np.abs(np.diff(signal_window))),
            np.sum(np.diff(np.sign(signal_window)) != 0) / max(n - 1, 1),
            (np.sum(np.diff(np.sign(np.diff(signal_window))) != 0) / max(n - 2, 1)
             if n > 2 else 0.0),
            np.trapz(abs_sig),
        ], dtype=np.float32)

    @staticmethod
    def _mean_only(signal: np.ndarray) -> np.ndarray:
        """Per-window mean subtraction (STEP 1, matching per_window_normalization='mean_only')."""
        return (signal - np.mean(signal)).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def process_window(
        self,
        raw_emg: np.ndarray,
        imu_data: np.ndarray,
        device_id: int = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Process one EMG window through the full pipeline and return the
        eight feature groups expected by EMGLSTMModel.

        Returns None only if the raw signal is completely invalid.

        Feature shapes (before normalisation):
            raw          (window_size,)
            rms_lms      (window_size,)
            wiener_td    (window_size,)
            wiener_fft   (64,)
            imu          (6, n_imu_samples)
            spectral_raw    (5,)
            spectral_wiener (5,)
            window_stats    (16,)
        """
        self._diagnostic_counter += 1

        # ── Ensure correct window size ────────────────────────────────────
        if len(raw_emg) != self.window_size:
            if len(raw_emg) < self.window_size:
                raw_emg = np.interp(
                    np.linspace(0, len(raw_emg) - 1, self.window_size),
                    np.arange(len(raw_emg)),
                    raw_emg,
                )
            else:
                raw_emg = raw_emg[: self.window_size]

        # ── IMU prep ──────────────────────────────────────────────────────
        if imu_data.ndim == 1:
            imu_data = imu_data.reshape(1, -1)
        if imu_data.shape[1] >= 6:
            gyro = imu_data[:, 3:6]
            gyro_mag = np.linalg.norm(gyro, axis=1)
        else:
            gyro_mag = np.zeros(len(raw_emg))

        if len(gyro_mag) != len(raw_emg):
            gyro_mag = (
                np.interp(
                    np.linspace(0, len(gyro_mag) - 1, len(raw_emg)),
                    np.arange(len(gyro_mag)),
                    gyro_mag,
                )
                if len(gyro_mag) > 1
                else np.full(len(raw_emg), gyro_mag[0] if len(gyro_mag) else 0.0)
            )

        # ── STEP 1: Bandpass filter ───────────────────────────────────────
        bp_filtered = np.array(
            [self._bandpass_filter(s) for s in raw_emg], dtype=np.float32
        )

        # ── STEP 2: LMS filter ────────────────────────────────────────────
        lms_filtered = np.zeros(self.window_size, dtype=np.float32)
        for i, (s, g) in enumerate(zip(bp_filtered, gyro_mag)):
            val = self._lms_filter_sample(float(s), float(g))
            self.lms_adaptation_count += 1
            if not np.isfinite(val):
                self.lms_weights[:] = 0.0
                self.lms_reference_buffer[:] = 0.0
                self.lms_adaptation_count = 0
                val = bp_filtered[i]
            lms_filtered[i] = (
                bp_filtered[i]
                if self.lms_adaptation_count < self.lms_min_adaptation_samples
                else val
            )

        # ── STEP 3: RMS envelope (single channel — RMS only) ─────────────
        rms_envelope = self._compute_rms_envelope(lms_filtered)   # (window_size,)

        # ── STEP 4: Wiener filter (time domain + FFT bins) ───────────────
        for s in bp_filtered:
            self.wiener_buffer.append(s)
            self.wiener_sample_count += 1

        if not self.wiener_is_calibrated and self.wiener_sample_count >= self.wiener_noise_samples:
            noise_data = np.array(list(self.wiener_buffer)[-self.wiener_noise_samples:])
            fft_noise = rfft(noise_data, n=self.wiener_fft_size)
            self.wiener_noise_power = np.maximum(np.abs(fft_noise) ** 2, 0.01)
            self.wiener_is_calibrated = True

        if len(self.wiener_buffer) >= self.wiener_fft_size:
            wiener_window = np.array(list(self.wiener_buffer)[-self.wiener_fft_size:])
        else:
            wiener_window = np.zeros(self.wiener_fft_size, dtype=np.float32)
            buf = list(self.wiener_buffer)
            wiener_window[-len(buf):] = buf

        wiener_td_full, wiener_fft_mags = self._wiener_filter(wiener_window)
        # wiener_td_full may be longer than window_size — trim/pad
        if len(wiener_td_full) >= self.window_size:
            wiener_td = wiener_td_full[: self.window_size]
        else:
            wiener_td = np.zeros(self.window_size, dtype=np.float32)
            wiener_td[: len(wiener_td_full)] = wiener_td_full

        # ── Bandpowers (for signal state check — use wiener FFT mags) ────
        emg_bandpowers = self._compute_bandpowers_from_mags(wiener_fft_mags)

        # ── STEP 5: Per-window normalisation (mean_only, matches training) ─
        raw_norm = self._mean_only(bp_filtered)
        rms_norm = self._mean_only(rms_envelope)
        wiener_td_norm = self._mean_only(wiener_td)
        wiener_fft_norm = self._mean_only(wiener_fft_mags)

        # ── STEP 6: Spectral features (from normalised windows) ───────────
        spectral_raw = self._compute_spectral_features(raw_norm)
        spectral_wiener = self._compute_spectral_features(wiener_td_norm)

        # ── STEP 7: Time-domain stats (16-d window_stats) ─────────────────
        stats_raw = self._compute_time_domain_stats(raw_norm)
        stats_wiener = self._compute_time_domain_stats(wiener_td_norm)
        window_stats = np.concatenate([stats_raw, stats_wiener])  # (16,)

        # ── STEP 8: IMU processing ────────────────────────────────────────
        # Determine expected IMU samples for this window (200 ms @ 100 Hz = 20)
        window_ms = (self.window_size / self.sampling_rate) * 1000.0
        n_imu_needed = max(1, int(window_ms / (1000.0 / self.imu_sampling_rate)))
        imu_window = self._prepare_imu_window(imu_data, n_imu_needed, device_id)
        # imu_window shape: (6, n_imu_needed)

        return {
            'raw':             raw_norm,
            'rms_lms':         rms_norm,
            'wiener_td':       wiener_td_norm,
            'wiener_fft':      wiener_fft_norm,
            'imu':             imu_window,
            'spectral_raw':    spectral_raw,
            'spectral_wiener': spectral_wiener,
            'window_stats':    window_stats,
            # Keep emg_bandpowers for signal-state gating (not fed to model)
            '_emg_bandpowers': emg_bandpowers,
            '_emg_envelope':   rms_norm,
        }

    def _compute_bandpowers_from_mags(self, fft_mags: np.ndarray) -> np.ndarray:
        """3-band bandpowers for signal-state gating (not fed to model)."""
        n = len(fft_mags)
        freqs = np.linspace(0, self.sampling_rate / 2, n)
        bp = np.array([
            np.sum(fft_mags[(freqs >= 20) & (freqs <= 60)] ** 2),
            np.sum(fft_mags[(freqs >= 60) & (freqs <= 150)] ** 2),
            np.sum(fft_mags[(freqs >= 150) & (freqs <= 250)] ** 2),
        ], dtype=np.float32)
        return bp

    def _prepare_imu_window(
        self, imu_data: np.ndarray, n_imu_needed: int, device_id: int
    ) -> np.ndarray:
        """
        Return (6, n_imu_needed) IMU array.

        Matches training:
          - Device 1 (LEFT): zero-out IMU (pure EMG arm)
          - All devices: per-time-step z-score across 6 channels,
            then transpose to (6, n_imu_samples)
        """
        if device_id == 1:
            return np.zeros((6, n_imu_needed), dtype=np.float32)

        # Ensure shape (n_samples, 6)
        if imu_data.shape[0] >= n_imu_needed:
            imu_w = imu_data[:n_imu_needed].copy()
        elif imu_data.shape[0] > 0:
            imu_w = np.zeros((n_imu_needed, 6), dtype=np.float32)
            imu_w[: imu_data.shape[0]] = imu_data
        else:
            return np.zeros((6, n_imu_needed), dtype=np.float32)

        # Per-time-step z-score across the 6 channels (matches training)
        for t in range(imu_w.shape[0]):
            row = imu_w[t]
            mean, std = np.mean(row), np.std(row)
            imu_w[t] = (row - mean) / (std + 1e-8)

        return imu_w.T.astype(np.float32)   # (6, n_imu_needed)

    # ──────────────────────────────────────────────────────────────────────
    # Normalisation (STEP 2: global StandardScaler)
    # ──────────────────────────────────────────────────────────────────────

    def normalize_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply global StandardScaler transforms (STEP 2).

        Scaler keys and expected dimensionality:
            raw, rms_lms, wiener_td  → 1-feature scaler (fitted on all samples)
            wiener_fft               → 64-feature scaler
            imu                      → 1-feature scaler (fitted on all flattened samples)
            spectral_raw,
            spectral_wiener          → 5-feature scaler
            window_stats             → 16-feature scaler
        """
        normalized = {}

        # 1-D signal scalers (raw, rms_lms, wiener_td)
        for key in ('raw', 'rms_lms', 'wiener_td'):
            sig = features[key]   # (window_size,)
            if key in self.scalers:
                sig = self.scalers[key].transform(sig.reshape(-1, 1)).flatten()
            normalized[key] = sig.astype(np.float32)

        # Per-window vector scalers
        for key, n_feat in (('wiener_fft', 64), ('spectral_raw', 5),
                            ('spectral_wiener', 5), ('window_stats', 16)):
            vec = features[key]   # (n_feat,)
            if key in self.scalers:
                vec = self.scalers[key].transform(vec.reshape(1, -1)).flatten()
            normalized[key] = vec.astype(np.float32)

        # IMU: 1-D scaler applied to all elements, reshape back
        imu = features['imu']   # (6, n_imu_samples)
        if 'imu' in self.scalers:
            flat = imu.flatten()
            flat = self.scalers['imu'].transform(flat.reshape(-1, 1)).flatten()
            imu = flat.reshape(imu.shape)
        normalized['imu'] = imu.astype(np.float32)

        return normalized

    # ──────────────────────────────────────────────────────────────────────
    # REST baseline (used by InferenceEngine for diagnostics / legacy)
    # ──────────────────────────────────────────────────────────────────────

    def collect_rest_baseline(self, features: Dict[str, np.ndarray]):
        if self.rest_baseline is None:
            self.rest_baseline = {
                'emg_envelope': {'values': []},
                'emg_bandpowers': {'values': []},
            }
        self.rest_baseline['emg_envelope']['values'].append(features['_emg_envelope'].flatten())
        self.rest_baseline['emg_bandpowers']['values'].append(features['_emg_bandpowers'])

    def finalize_rest_baseline(self):
        if self.rest_baseline is None or len(self.rest_baseline['emg_envelope']['values']) == 0:
            self.rest_baseline = {
                'emg_envelope':   {'values': [], 'mean': 5.0, 'std': 3.0},
                'emg_bandpowers': {'values': [], 'mean': np.array([5., 10., 5.]),
                                   'std': np.array([2., 3., 2.])},
            }
            self.rest_baseline_collected = True
            return True

        env_all = np.concatenate(self.rest_baseline['emg_envelope']['values'])
        med = np.median(env_all)
        mad = np.median(np.abs(env_all - med))
        self.rest_baseline['emg_envelope']['mean'] = med
        self.rest_baseline['emg_envelope']['std'] = max(1.4826 * mad, 1.0) + 1e-10

        bp_all = np.stack(self.rest_baseline['emg_bandpowers']['values'])
        bp_med = np.median(bp_all, axis=0)
        bp_mad = np.median(np.abs(bp_all - bp_med), axis=0)
        self.rest_baseline['emg_bandpowers']['mean'] = bp_med
        self.rest_baseline['emg_bandpowers']['std'] = np.maximum(1.4826 * bp_mad, 1.0) + 1e-10

        self.rest_baseline_collected = True
        self.rest_baseline['emg_envelope']['values'] = []
        self.rest_baseline['emg_bandpowers']['values'] = []
        return True
