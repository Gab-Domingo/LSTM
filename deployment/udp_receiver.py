#!/usr/bin/env python3
"""
UDP Receiver for Real-Time EMG Gesture Recognition
==================================================
Optimized for Raspberry Pi with simple terminal logging.
Only logs when gesture recognition changes.
"""

import socket
import json
import numpy as np
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from scipy.signal import butter, sosfilt
from scipy.fft import fft, irfft

from inference import (
    load_model,
    load_scalers,
    create_window_from_data,
    create_sequence_from_windows,
    predict_gesture,
    SimpleLogger
)

# =============================================================
# FILTER IMPLEMENTATIONS
# =============================================================

class BandpassFilter:
    """Bandpass filter (20-200 Hz) for EMG"""
    
    def __init__(self, lowcut=20, highcut=200, fs=500, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if high >= 1.0:
            high = 0.95
        self.sos = butter(order, [low, high], btype="band", output='sos')
        self.zi = np.zeros((self.sos.shape[0], 2))
    
    def process(self, sample):
        filtered, self.zi = sosfilt(self.sos, [sample], zi=self.zi)
        return filtered[0]


class LMSAdaptiveFilter:
    """LMS Adaptive Filter for motion artifact removal"""
    
    def __init__(self, filter_length=16, mu=0.0002):
        self.filter_length = filter_length
        self.mu = mu
        self.weights = np.zeros(filter_length)
        self.reference_buffer = np.zeros(filter_length)
        self.emg_mean = 2048.0
        self.emg_var = 250000.0
        self.gyro_var = 10000.0
        self.alpha = 0.001
    
    def update_statistics(self, emg_raw, gyro_raw):
        emg_centered = emg_raw - self.emg_mean
        self.emg_mean = (1 - self.alpha) * self.emg_mean + self.alpha * emg_raw
        self.emg_var = (1 - self.alpha) * self.emg_var + self.alpha * (emg_centered ** 2)
        self.gyro_var = (1 - self.alpha) * self.gyro_var + self.alpha * (gyro_raw ** 2)
        return emg_centered
    
    def normalize(self, emg_centered, gyro_raw):
        emg_std = np.sqrt(self.emg_var + 1e-8)
        gyro_std = np.sqrt(self.gyro_var + 1e-8)
        emg_norm = emg_centered / emg_std
        gyro_norm = gyro_raw / gyro_std
        return emg_norm, gyro_norm, emg_std
    
    def filter_sample(self, emg_sample, gyro_mag):
        emg_centered = self.update_statistics(emg_sample, gyro_mag)
        emg_norm, gyro_norm, emg_std = self.normalize(emg_centered, gyro_mag)
        
        self.reference_buffer = np.roll(self.reference_buffer, 1)
        self.reference_buffer[0] = gyro_norm
        
        estimated_artifact = np.dot(self.weights, self.reference_buffer)
        error = emg_norm - estimated_artifact
        
        reference_power = np.dot(self.reference_buffer, self.reference_buffer)
        
        if reference_power > 1e-6:
            normalized_mu = self.mu / (reference_power + 0.1)
            weight_delta = normalized_mu * error * self.reference_buffer
            weight_delta = np.clip(weight_delta, -0.01, 0.01)
            self.weights += weight_delta
            self.weights = np.clip(self.weights, -5.0, 5.0)
        
        filtered_emg_centered = error * emg_std
        filtered_emg = filtered_emg_centered + self.emg_mean
        
        if not np.isfinite(filtered_emg):
            self.weights = np.zeros(self.filter_length)
            self.reference_buffer = np.zeros(self.filter_length)
            return emg_sample
        
        return filtered_emg


class RMSFilter:
    """RMS filter with sliding window"""
    
    def __init__(self, window_size=50, overlap=0.5):
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        self.buffer = deque(maxlen=window_size)
        self.sample_count = 0
    
    def process(self, sample):
        self.buffer.append(sample)
        self.sample_count += 1
        
        if len(self.buffer) == self.window_size and self.sample_count % self.stride == 0:
            values = np.array(list(self.buffer))
            rms = np.sqrt(np.mean(values ** 2))
            return rms
        return None


class WienerFilter:
    """Wiener filter in frequency domain"""
    
    def __init__(self, fft_size=64, snr_threshold=2.0, noise_estimation_samples=500):
        self.fft_size = fft_size
        self.snr_threshold = snr_threshold
        self.noise_estimation_samples = noise_estimation_samples
        self.noise_power = None
        self.is_calibrated = False
        self.sample_count = 0
        self.emg_buffer = deque(maxlen=fft_size * 2)
    
    def process(self, sample):
        self.emg_buffer.append(sample)
        self.sample_count += 1
        
        if not self.is_calibrated:
            if self.sample_count >= self.noise_estimation_samples:
                self._calibrate_noise()
            return sample, None
        
        if len(self.emg_buffer) >= self.fft_size:
            filtered, fft_mags = self._wiener_filter_fft()
            return filtered, fft_mags
        
        return sample, None
    
    def _calibrate_noise(self):
        noise_data = np.array(list(self.emg_buffer)[-self.noise_estimation_samples:])
        fft_data = np.fft.rfft(noise_data, n=self.fft_size)
        self.noise_power = np.abs(fft_data) ** 2
        self.is_calibrated = True
    
    def _wiener_filter_fft(self):
        window = np.array(list(self.emg_buffer)[-self.fft_size:])
        fft_signal = np.fft.rfft(window, n=self.fft_size)
        signal_power = np.abs(fft_signal) ** 2
        
        wiener_gain = np.maximum(0, 1 - (self.noise_power / (signal_power + 1e-10)))
        snr = signal_power / (self.noise_power + 1e-10)
        wiener_gain = np.where(snr > self.snr_threshold, wiener_gain, 0.0)
        
        filtered_fft = fft_signal * wiener_gain
        filtered_signal = np.fft.irfft(filtered_fft, n=self.fft_size)
        
        fft_magnitudes = np.abs(fft_signal[:self.fft_size // 2])
        if len(fft_magnitudes) < 64:
            fft_magnitudes = np.interp(
                np.linspace(0, len(fft_magnitudes)-1, 64),
                np.arange(len(fft_magnitudes)), fft_magnitudes
            )
        
        return filtered_signal[-1], fft_magnitudes


# =============================================================
# UDP RECEIVER AND PROCESSING
# =============================================================

class UDPEMGReceiver:
    """UDP receiver for EMG data with real-time filtering and inference"""
    
    def __init__(self, model_path: str, config_path: str, scaler_path: Optional[str] = None,
                 host: str = '0.0.0.0', port: int = 5000,
                 device: str = 'cpu', debug: bool = False):
        # Load model
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model, _, class_names_raw = load_model(model_path, device=device)
        
        # Convert class names to strings
        if class_names_raw and isinstance(class_names_raw[0], (int, np.integer)):
            self.class_names = ['REST', 'FIST']
        else:
            self.class_names = [str(name) for name in class_names_raw]
        
        # Load StandardScalers for two-step normalization
        self.scalers = None
        if scaler_path:
            scaler_file = Path(scaler_path)
            if scaler_file.exists():
                self.scalers = load_scalers(scaler_file)
                print(f"✓ StandardScalers loaded")
            else:
                print(f"⚠️  StandardScalers not found: {scaler_path}")
        
        self.threshold = self.config.get('optimal_threshold', 0.70)
        self.window_size = self.config['window_size']
        self.sequence_length = self.config['sequence_length']
        self.stride = self.config.get('stride', 25)  # Window stride for real-time
        self.sampling_rate = self.config.get('sampling_rate', 500)
        
        # Initialize filters
        self.bandpass = BandpassFilter()
        self.lms = LMSAdaptiveFilter()
        self.rms_filter = RMSFilter(window_size=50, overlap=0.5)
        self.wiener = WienerFilter(fft_size=64)
        
        # Buffers - need larger buffers for stride-based windowing
        self.raw_buffer = deque(maxlen=self.window_size + self.stride * 2)
        self.rms_buffer = deque(maxlen=self.window_size + self.stride * 2)
        self.lms_buffer = deque(maxlen=self.window_size + self.stride * 2)
        self.wiener_td_buffer = deque(maxlen=self.window_size + self.stride * 2)
        self.imu_buffer = deque(maxlen=20)
        self.window_buffer = deque(maxlen=self.sequence_length)
        
        # IMU downsampling (500Hz -> 100Hz)
        self.imu_sample_counter = 0
        self.imu_downsample_factor = 5
        
        # Current Wiener FFT
        self.current_wiener_fft = np.zeros(64, dtype=np.float32)
        
        # Statistics
        self.total_samples = 0
        self.total_predictions = 0
        self.last_prediction_time = time.time()
        self.last_window_sample = 0  # Track last sample used for window creation
        self.prediction_times = deque(maxlen=10)  # Track last 10 prediction times for FPS calculation
        
        # Temporal smoothing - reduced for real-time (optional, can disable)
        self.prediction_history = deque(maxlen=3)  # Reduced from 5 to 3
        self.smoothed_prediction = None
        self.use_temporal_smoothing = True  # Can disable for fastest response
        
        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.sock.bind((host, port))
        self.sock.settimeout(1.0)
        
        # Buffer for incomplete JSON packets
        self.json_buffer = b''
        self.packet_count = 0
        self.error_count = 0
        self.debug = debug
        
        print(f"✓ UDP receiver initialized on {host}:{port}")
        print(f"  Model threshold: {self.threshold:.2f}")
        print(f"  Window size: {self.window_size} samples ({self.window_size*2}ms @ 500Hz)")
        print(f"  Stride: {self.stride} samples ({self.stride*2}ms - predictions every {self.stride*2}ms)")
        print(f"  Sequence length: {self.sequence_length} windows")
        print(f"  Initial delay: ~{self.window_size + (self.sequence_length-1)*self.stride} samples ({((self.window_size + (self.sequence_length-1)*self.stride)*2)}ms)")
        print(f"  Classes: {self.class_names}")
        print(f"  Temporal smoothing: {'ON' if self.use_temporal_smoothing else 'OFF'}")
    
    def parse_udp_packet(self, data: bytes) -> List[Dict]:
        """Parse JSON batch from UDP packet - handles packet drops gracefully"""
        self.packet_count += 1
        self.json_buffer += data
        
        batches = []
        
        # More aggressive buffer size check - reset earlier to prevent memory issues
        MAX_BUFFER_SIZE = 50000  # Reduced from 100000 to catch issues earlier
        if len(self.json_buffer) > MAX_BUFFER_SIZE:
            if self.debug:
                print(f"⚠️ Buffer too large ({len(self.json_buffer)} bytes), resetting...")
            self.json_buffer = b''
            self.error_count += 1
            return []
        
        try:
            json_str = self.json_buffer.decode('utf-8', errors='ignore')
            
            # Try to find complete JSON arrays more aggressively
            bracket_count = 0
            start_idx = -1
            last_valid_end = -1
            
            for i, char in enumerate(json_str):
                if char == '[':
                    if bracket_count == 0:
                        start_idx = i
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0 and start_idx >= 0:
                        last_valid_end = i
                        try:
                            json_obj = json_str[start_idx:i+1]
                            batch = json.loads(json_obj)
                            if isinstance(batch, list):
                                batches.append(batch)
                                # Clear buffer up to this point
                                self.json_buffer = self.json_buffer[i+1:]
                                break
                        except json.JSONDecodeError:
                            continue
            
            # If we found a valid end but didn't parse, try to extract it
            if not batches and last_valid_end > 0:
                try:
                    json_obj = json_str[start_idx:last_valid_end+1]
                    batch = json.loads(json_obj)
                    if isinstance(batch, list):
                        batches.append(batch)
                        self.json_buffer = self.json_buffer[last_valid_end+1:]
                except json.JSONDecodeError:
                    pass
            
            # If still no batches, try parsing entire buffer
            if not batches:
                try:
                    batch = json.loads(json_str.strip())
                    if isinstance(batch, list):
                        batches.append(batch)
                        self.json_buffer = b''
                except json.JSONDecodeError:
                    # Suppress ALL warnings - JSON parse failures are normal with UDP fragmentation
                    # Only attempt recovery if buffer is getting large (and only log in debug mode)
                    if len(self.json_buffer) > 10000:
                        # Try to find any valid JSON in buffer
                        json_str = self.json_buffer.decode('utf-8', errors='ignore')
                        start = json_str.find('[')
                        end = json_str.rfind(']')
                        if start >= 0 and end > start:
                            try:
                                json_obj = json_str[start:end+1]
                                batch = json.loads(json_obj)
                                if isinstance(batch, list):
                                    batches.append(batch)
                                    self.json_buffer = self.json_buffer[end+1:]
                            except:
                                # If recovery fails, clear buffer to prevent infinite growth
                                # Suppress warnings - only show every 50th failure in debug mode
                                if self.debug and self.error_count % 50 == 0:
                                    print(f"⚠️ Recovery failed, clearing buffer ({len(self.json_buffer)} bytes)")
                                self.json_buffer = b''
                                self.error_count += 1
                    # For any buffer size, just return empty silently - don't spam warnings
                    # JSON parse failures are normal with UDP fragmentation/packet drops
                    return []
        
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self.error_count += 1
            # Only print errors in debug mode or every 100th error (UDP drops are normal)
            if self.debug or self.error_count % 100 == 1:
                print(f"⚠️ JSON parse error (count: {self.error_count}): {str(e)[:100]}")
            
            # More aggressive recovery
            if len(self.json_buffer) > 5000:
                try:
                    json_str = self.json_buffer.decode('utf-8', errors='ignore')
                    start = json_str.find('[')
                    end = json_str.rfind(']')
                    if start >= 0 and end > start:
                        try:
                            json_obj = json_str[start:end+1]
                            batch = json.loads(json_obj)
                            if isinstance(batch, list):
                                batches.append(batch)
                                self.json_buffer = self.json_buffer[end+1:]
                        except:
                            # Clear buffer if recovery fails
                            # Suppress warnings - only show every 50th failure
                            if self.debug and self.error_count % 50 == 0:
                                print(f"⚠️ Recovery failed, clearing buffer")
                            self.json_buffer = b''
                except:
                    # If all else fails, clear buffer
                    if self.debug:
                        print(f"⚠️ Critical error, clearing buffer")
                    self.json_buffer = b''
            
            return []
        
        all_samples = []
        for batch in batches:
            if isinstance(batch, list):
                all_samples.extend(batch)
        
        return all_samples
    
    def process_sample(self, sample: Dict) -> Optional[Tuple]:
        """Process a single sample through filters"""
        raw_emg = float(sample['m'])
        ax, ay, az = float(sample['ax']), float(sample['ay']), float(sample['az'])
        gx, gy, gz = float(sample['gx']), float(sample['gy']), float(sample['gz'])
        
        # Apply filters
        bandpass_emg = self.bandpass.process(raw_emg)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        lms_emg = self.lms.filter_sample(bandpass_emg, gyro_mag)
        rms_result = self.rms_filter.process(lms_emg)
        wiener_td, wiener_fft = self.wiener.process(bandpass_emg)
        
        if wiener_fft is not None:
            wiener_fft = np.asarray(wiener_fft, dtype=np.float32).flatten()
            if len(wiener_fft) == 64:
                self.current_wiener_fft = wiener_fft
            elif len(wiener_fft) > 64:
                self.current_wiener_fft = wiener_fft[:64]
            else:
                padded = np.zeros(64, dtype=np.float32)
                padded[:len(wiener_fft)] = wiener_fft
                self.current_wiener_fft = padded
        
        # Add to buffers
        self.raw_buffer.append(raw_emg)
        self.lms_buffer.append(lms_emg)
        if rms_result is not None:
            self.rms_buffer.append(rms_result)
        else:
            if len(self.rms_buffer) > 0:
                self.rms_buffer.append(self.rms_buffer[-1])
            else:
                self.rms_buffer.append(abs(lms_emg))
        
        self.wiener_td_buffer.append(wiener_td)
        
        # IMU downsampling
        self.imu_sample_counter += 1
        if self.imu_sample_counter >= self.imu_downsample_factor:
            self.imu_sample_counter = 0
            imu_sample = np.array([ax, ay, az, gx, gy, gz], dtype=np.float32)
            self.imu_buffer.append(imu_sample)
        
        self.total_samples += 1
        
        # Create windows every stride samples for real-time performance
        # This allows predictions every 50ms (stride=25 @ 500Hz) instead of every 200ms
        if len(self.raw_buffer) >= self.window_size:
            # Check if we should create a new window (every stride samples)
            samples_since_last_window = self.total_samples - self.last_window_sample
            if samples_since_last_window >= self.stride:
                self.last_window_sample = self.total_samples
                return self._create_prediction()
        
        return None
    
    def _create_prediction(self) -> Optional[Tuple]:
        """Create window and run inference"""
        # Convert buffers to arrays
        raw_array = np.array(list(self.raw_buffer)[-self.window_size:])
        lms_array = np.array(list(self.lms_buffer)[-self.window_size:])
        
        rms_list = list(self.rms_buffer)
        if len(rms_list) < self.window_size:
            rms_array = np.array(rms_list + [rms_list[-1]] * (self.window_size - len(rms_list)))
        else:
            rms_array = np.array(rms_list[-self.window_size:])
        
        wiener_td_array = np.array(list(self.wiener_td_buffer)[-self.window_size:])
        
        imu_list = list(self.imu_buffer)
        if len(imu_list) < 10:
            if len(imu_list) > 0:
                imu_array = np.array(imu_list + [imu_list[-1]] * (10 - len(imu_list)))
            else:
                imu_array = np.zeros((10, 6), dtype=np.float32)
        else:
            imu_array = np.array(imu_list[-10:])
        
        # Create window using inference.py function (with two-step normalization)
        window = create_window_from_data(
            raw_array, rms_array, lms_array, wiener_td_array,
            self.current_wiener_fft, imu_array,
            0, self.window_size, self.sampling_rate, self.scalers
        )
        
        self.window_buffer.append(window)
        
        # Run inference when we have enough windows
        if len(self.window_buffer) >= self.sequence_length:
            sequence = create_sequence_from_windows(
                list(self.window_buffer), self.sequence_length
            )
            
            # Time the inference
            inference_start = time.time()
            prediction, confidence, probs = predict_gesture(
                self.model, sequence, self.threshold, 'cpu'
            )
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            self.total_predictions += 1
            current_time = time.time()
            self.prediction_times.append(current_time)
            
            # Calculate FPS from recent predictions
            if len(self.prediction_times) >= 2:
                time_diff = self.prediction_times[-1] - self.prediction_times[0]
                fps = (len(self.prediction_times) - 1) / time_diff if time_diff > 0 else 0
            else:
                fps = 0
            
            # Debug output (only in debug mode to avoid spam)
            if self.debug and self.total_predictions % 20 == 0:
                print(f"\n[DEBUG] Prediction #{self.total_predictions}: "
                      f"inference={inference_time:.1f}ms, fps={fps:.1f}Hz", flush=True)
            
            return prediction, confidence, probs
        
        return None
    
    def _apply_temporal_smoothing(self, prediction: int, confidence: float, probs: np.ndarray) -> Tuple[int, float]:
        """Apply lightweight temporal smoothing - can be disabled for fastest response"""
        if not self.use_temporal_smoothing:
            return prediction, confidence
        
        self.prediction_history.append({
            'prediction': prediction,
            'confidence': confidence,
            'probs': probs.copy()
        })
        
        # Reduced smoothing window for real-time (only 2 predictions needed)
        if len(self.prediction_history) < 2:
            self.smoothed_prediction = prediction
            return prediction, confidence
        
        recent = list(self.prediction_history)[-2:]  # Only last 2 predictions
        predictions = [p['prediction'] for p in recent]
        
        # If both agree, use it immediately (no delay)
        if len(set(predictions)) == 1:
            self.smoothed_prediction = predictions[0]
            return predictions[0], confidence
        # If they disagree but confidence is high, trust current
        elif confidence > 0.70:
            self.smoothed_prediction = prediction
            return prediction, confidence
        # Otherwise keep previous
        elif self.smoothed_prediction is not None:
            return self.smoothed_prediction, confidence
        else:
            self.smoothed_prediction = prediction
            return prediction, confidence
    
    def run(self, logger: Optional[SimpleLogger] = None):
        """Main loop: receive UDP packets and process"""
        print("\n" + "="*60)
        print("Starting UDP receiver...")
        print("Waiting for data from ESP32 device...")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        if logger is None:
            logger = SimpleLogger()
        
        try:
            while True:
                try:
                    data, addr = self.sock.recvfrom(65536)
                    batch = self.parse_udp_packet(data)
                    
                    if not batch:
                        continue
                    
                    valid_samples = []
                    for sample in batch:
                        if isinstance(sample, dict) and 'm' in sample:
                            valid_samples.append(sample)
                    
                    if not valid_samples:
                        continue
                    
                    # Process each sample
                    for sample in valid_samples:
                        result = self.process_sample(sample)
                        
                        if result is not None:
                            prediction, confidence, probs = result
                            
                            # Apply temporal smoothing
                            smoothed_prediction, smoothed_confidence = self._apply_temporal_smoothing(
                                prediction, confidence, probs
                            )
                            gesture_name = self.class_names[smoothed_prediction]
                            
                            # Log only when gesture changes
                            logger.log(gesture_name, smoothed_confidence, probs)
                
                except socket.timeout:
                    continue
                
                except KeyboardInterrupt:
                    break
        
        except KeyboardInterrupt:
            print("\n\nStopping receiver...")
        
        finally:
            self.sock.close()
            print(f"\n✓ Statistics:")
            print(f"  Total packets: {self.packet_count}")
            print(f"  Parse errors: {self.error_count} ({100*self.error_count/max(self.packet_count,1):.1f}%)")
            print(f"  Total samples: {self.total_samples}")
            print(f"  Total predictions: {self.total_predictions}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='UDP Receiver for EMG Gesture Recognition')
    parser.add_argument('--model', type=str, default='emg_lstm_model.pt',
                       help='Path to model file')
    parser.add_argument('--config', type=str, default='deployment_config.json',
                       help='Path to config file')
    parser.add_argument('--scalers', type=str, default='standard_scalers.pkl',
                       help='Path to StandardScalers file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='UDP host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='UDP port (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable temporal smoothing for fastest response (may be noisier)')
    
    args = parser.parse_args()
    
    receiver = UDPEMGReceiver(
        model_path=args.model,
        config_path=args.config,
        scaler_path=args.scalers,
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    # Disable temporal smoothing if requested
    if args.no_smoothing:
        receiver.use_temporal_smoothing = False
        print("⚠️  Temporal smoothing DISABLED - fastest response mode")
    
    receiver.run()


if __name__ == '__main__':
    main()
