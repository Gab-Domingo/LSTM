"""
Enhanced feature monitoring for real-time EMG analysis.
Tracks amplitude and frequency domain features to help distinguish REST vs FIST.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Optional, List
from datetime import datetime


class FeatureMonitor:
    """
    Real-time feature monitoring for EMG signals.
    Tracks amplitude (envelope) and frequency domain (bandpowers) features.
    """
    
    def __init__(self, history_size: int = 50):
        """
        Args:
            history_size: Number of feature samples to keep in history
        """
        self.history_size = history_size
        
        # Feature history
        self.envelope_history = deque(maxlen=history_size)
        self.bandpowers_history = deque(maxlen=history_size)
        self.imu_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
        # REST baseline reference (for comparison)
        self.rest_baseline = {
            'envelope_mean': None,
            'envelope_std': None,
            'bandpowers_mean': None,
            'bandpowers_std': None
        }
        
        # Color codes
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.CYAN = '\033[96m'
        self.MAGENTA = '\033[95m'
        self.BOLD = '\033[1m'
        self.RESET = '\033[0m'
    
    def set_rest_baseline(self, envelope_mean: float, envelope_std: float,
                         bandpowers_mean: np.ndarray, bandpowers_std: np.ndarray):
        """
        Set REST baseline for comparison.
        
        Args:
            envelope_mean: REST envelope mean
            envelope_std: REST envelope std
            bandpowers_mean: REST bandpowers mean (3,)
            bandpowers_std: REST bandpowers std (3,)
        """
        self.rest_baseline['envelope_mean'] = envelope_mean
        self.rest_baseline['envelope_std'] = envelope_std
        self.rest_baseline['bandpowers_mean'] = bandpowers_mean.copy()
        self.rest_baseline['bandpowers_std'] = bandpowers_std.copy()
    
    def add_features(self, envelope: np.ndarray, bandpowers: np.ndarray, 
                    imu_features: np.ndarray):
        """
        Add feature sample to history.
        
        Args:
            envelope: EMG envelope (window_size,)
            bandpowers: Bandpowers [low, mid, high] (3,)
            imu_features: IMU features [gyro_mag_mean, accel_mag_mean, gyro_mag_std, accel_mag_std] (4,)
        """
        timestamp = time.time()
        
        # Store summary statistics
        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        envelope_max = np.max(envelope)
        
        self.envelope_history.append({
            'mean': envelope_mean,
            'std': envelope_std,
            'max': envelope_max,
            'timestamp': timestamp
        })
        
        self.bandpowers_history.append({
            'low': bandpowers[0],
            'mid': bandpowers[1],
            'high': bandpowers[2],
            'total': np.sum(bandpowers),
            'timestamp': timestamp
        })
        
        self.imu_history.append({
            'gyro_mag_mean': imu_features[0],
            'accel_mag_mean': imu_features[1],
            'gyro_mag_std': imu_features[2],
            'accel_mag_std': imu_features[3],
            'timestamp': timestamp
        })
        
        self.timestamps.append(timestamp)
    
    def get_envelope_stats(self, window: int = 10) -> Dict:
        """Get envelope statistics over recent window."""
        if len(self.envelope_history) == 0:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'vs_rest': 0.0}
        
        recent = list(self.envelope_history)[-window:]
        mean = np.mean([e['mean'] for e in recent])
        std = np.mean([e['std'] for e in recent])
        max_val = np.mean([e['max'] for e in recent])
        
        # Compare to REST baseline
        vs_rest = 0.0
        if self.rest_baseline['envelope_mean'] is not None:
            vs_rest = (mean - self.rest_baseline['envelope_mean']) / (
                self.rest_baseline['envelope_std'] + 1e-10
            )
        
        return {
            'mean': mean,
            'std': std,
            'max': max_val,
            'vs_rest': vs_rest
        }
    
    def get_bandpower_stats(self, window: int = 10) -> Dict:
        """Get bandpower statistics over recent window."""
        if len(self.bandpowers_history) == 0:
            return {
                'low': 0.0, 'mid': 0.0, 'high': 0.0, 'total': 0.0,
                'vs_rest_low': 0.0, 'vs_rest_mid': 0.0, 'vs_rest_high': 0.0
            }
        
        recent = list(self.bandpowers_history)[-window:]
        low = np.mean([b['low'] for b in recent])
        mid = np.mean([b['mid'] for b in recent])
        high = np.mean([b['high'] for b in recent])
        total = np.mean([b['total'] for b in recent])
        
        # Compare to REST baseline
        vs_rest_low = vs_rest_mid = vs_rest_high = 0.0
        if self.rest_baseline['bandpowers_mean'] is not None:
            rest_mean = self.rest_baseline['bandpowers_mean']
            rest_std = self.rest_baseline['bandpowers_std']
            vs_rest_low = (low - rest_mean[0]) / (rest_std[0] + 1e-10)
            vs_rest_mid = (mid - rest_mean[1]) / (rest_std[1] + 1e-10)
            vs_rest_high = (high - rest_mean[2]) / (rest_std[2] + 1e-10)
        
        return {
            'low': low,
            'mid': mid,
            'high': high,
            'total': total,
            'vs_rest_low': vs_rest_low,
            'vs_rest_mid': vs_rest_mid,
            'vs_rest_high': vs_rest_high
        }
    
    def get_feature_trends(self, window: int = 20) -> Dict:
        """Get feature trends (increasing/decreasing)."""
        if len(self.envelope_history) < window:
            return {'envelope': '?', 'bandpower_total': '?', 'bandpower_high': '?'}
        
        recent_env = list(self.envelope_history)[-window:]
        recent_bp = list(self.bandpowers_history)[-window:]
        
        # Envelope trend
        env_first = np.mean([e['mean'] for e in recent_env[:window//2]])
        env_second = np.mean([e['mean'] for e in recent_env[window//2:]])
        env_trend = '↑' if env_second > env_first * 1.1 else ('↓' if env_second < env_first * 0.9 else '→')
        
        # Bandpower total trend
        bp_first = np.mean([b['total'] for b in recent_bp[:window//2]])
        bp_second = np.mean([b['total'] for b in recent_bp[window//2:]])
        bp_trend = '↑' if bp_second > bp_first * 1.1 else ('↓' if bp_second < bp_first * 0.9 else '→')
        
        # High frequency bandpower trend (important for FIST detection)
        bp_high_first = np.mean([b['high'] for b in recent_bp[:window//2]])
        bp_high_second = np.mean([b['high'] for b in recent_bp[window//2:]])
        bp_high_trend = '↑' if bp_high_second > bp_high_first * 1.1 else ('↓' if bp_high_second < bp_high_first * 0.9 else '→')
        
        return {
            'envelope': env_trend,
            'bandpower_total': bp_trend,
            'bandpower_high': bp_high_trend
        }
    
    def print_feature_dashboard(self, current_prediction: str, confidence: float):
        """
        Print detailed feature dashboard.
        
        Args:
            current_prediction: Current predicted class
            confidence: Current confidence
        """
        if len(self.envelope_history) == 0:
            print(f"\n{self.YELLOW}⏳ Waiting for features...{self.RESET}")
            return
        
        print("\n" + "="*70)
        print(f"{self.BOLD}📊 Real-Time Feature Monitor{self.RESET}")
        print("="*70)
        
        # Current prediction
        pred_color = self.GREEN if current_prediction == 'FIST' else self.BLUE
        conf_color = self.GREEN if confidence >= 0.8 else (self.YELLOW if confidence >= 0.6 else self.RED)
        symbol = '✊' if current_prediction == 'FIST' else '🛑'
        print(f"\n{self.BOLD}Current: {pred_color}{symbol} {current_prediction}{self.RESET} "
              f"({conf_color}{confidence:.1%}{self.RESET})")
        
        # Envelope (Amplitude) Features
        env_stats = self.get_envelope_stats(window=10)
        print(f"\n{self.BOLD}📈 Amplitude (EMG Envelope):{self.RESET}")
        print(f"   Mean: {self._format_value(env_stats['mean'], 2)}")
        print(f"   Max:  {self._format_value(env_stats['max'], 2)}")
        print(f"   Std:  {self._format_value(env_stats['std'], 2)}")
        
        if self.rest_baseline['envelope_mean'] is not None:
            vs_rest = env_stats['vs_rest']
            vs_color = self.GREEN if vs_rest > 1.0 else (self.YELLOW if vs_rest > 0.5 else self.BLUE)
            print(f"   vs REST: {vs_color}{vs_rest:+.2f}σ{self.RESET} "
                  f"(REST baseline: {self.rest_baseline['envelope_mean']:.1f})")
        
        # Frequency Domain (Bandpowers)
        bp_stats = self.get_bandpower_stats(window=10)
        print(f"\n{self.BOLD}🎵 Frequency Domain (Bandpowers):{self.RESET}")
        print(f"   Low (20-60Hz):   {self._format_value(bp_stats['low'], 1)}")
        print(f"   Mid (60-150Hz):  {self._format_value(bp_stats['mid'], 1)}")
        print(f"   High (150-250Hz): {self._format_value(bp_stats['high'], 1)}")
        print(f"   Total:            {self._format_value(bp_stats['total'], 1)}")
        
        if self.rest_baseline['bandpowers_mean'] is not None:
            print(f"   {self.BOLD}vs REST baseline:{self.RESET}")
            vs_low = bp_stats['vs_rest_low']
            vs_mid = bp_stats['vs_rest_mid']
            vs_high = bp_stats['vs_rest_high']
            print(f"      Low:   {self._color_vs_rest(vs_low)}{vs_low:+.2f}σ{self.RESET}")
            print(f"      Mid:   {self._color_vs_rest(vs_mid)}{vs_mid:+.2f}σ{self.RESET}")
            print(f"      High:  {self._color_vs_rest(vs_high)}{vs_high:+.2f}σ{self.RESET} "
                  f"{self.CYAN}(High freq is key for FIST){self.RESET}")
        
        # Feature Trends
        trends = self.get_feature_trends(window=20)
        print(f"\n{self.BOLD}📉 Trends (last 20 samples):{self.RESET}")
        print(f"   Envelope:     {self._color_trend(trends['envelope'])}")
        print(f"   Bandpower:    {self._color_trend(trends['bandpower_total'])}")
        print(f"   High Freq:    {self._color_trend(trends['bandpower_high'])}")
        
        # Feature Analysis
        print(f"\n{self.BOLD}🔍 Feature Analysis:{self.RESET}")
        analysis = self._analyze_features(env_stats, bp_stats)
        for key, value in analysis.items():
            print(f"   {key}: {value}")
        
        print("="*70)
    
    def _format_value(self, value: float, decimals: int = 1) -> str:
        """Format value with appropriate color."""
        if value > 1000:
            return f"{value/1000:.{decimals}f}k"
        return f"{value:.{decimals}f}"
    
    def _color_vs_rest(self, vs_rest: float) -> str:
        """Color code vs REST baseline."""
        if vs_rest > 2.0:
            return self.GREEN
        elif vs_rest > 1.0:
            return self.YELLOW
        elif vs_rest > 0.0:
            return self.CYAN
        else:
            return self.BLUE
    
    def _color_trend(self, trend: str) -> str:
        """Color code trend."""
        if trend == '↑':
            return f"{self.GREEN}↑ Increasing{self.RESET}"
        elif trend == '↓':
            return f"{self.RED}↓ Decreasing{self.RESET}"
        else:
            return f"{self.YELLOW}→ Stable{self.RESET}"
    
    def _analyze_features(self, env_stats: Dict, bp_stats: Dict) -> Dict:
        """Analyze features to provide insights."""
        analysis = {}
        
        # Amplitude analysis
        if self.rest_baseline['envelope_mean'] is not None:
            vs_rest = env_stats['vs_rest']
            if vs_rest > 2.0:
                analysis['Amplitude'] = f"{self.GREEN}Strong FIST signal (>2σ above REST){self.RESET}"
            elif vs_rest > 1.0:
                analysis['Amplitude'] = f"{self.YELLOW}Moderate FIST signal (>1σ above REST){self.RESET}"
            elif vs_rest > 0.0:
                analysis['Amplitude'] = f"{self.CYAN}Weak FIST signal (slightly above REST){self.RESET}"
            else:
                analysis['Amplitude'] = f"{self.BLUE}REST-like amplitude{self.RESET}"
        
        # Frequency analysis
        if self.rest_baseline['bandpowers_mean'] is not None:
            vs_high = bp_stats['vs_rest_high']
            if vs_high > 1.5:
                analysis['High Freq'] = f"{self.GREEN}Strong high-frequency content (FIST indicator){self.RESET}"
            elif vs_high > 0.5:
                analysis['High Freq'] = f"{self.YELLOW}Moderate high-frequency content{self.RESET}"
            else:
                analysis['High Freq'] = f"{self.BLUE}Low high-frequency content (REST-like){self.RESET}"
        
        # Combined analysis
        if self.rest_baseline['envelope_mean'] is not None and self.rest_baseline['bandpowers_mean'] is not None:
            env_vs = env_stats['vs_rest']
            high_vs = bp_stats['vs_rest_high']
            
            if env_vs > 1.0 and high_vs > 1.0:
                analysis['Overall'] = f"{self.GREEN}✓ Clear FIST pattern (amplitude + frequency){self.RESET}"
            elif env_vs > 1.0 or high_vs > 1.0:
                analysis['Overall'] = f"{self.YELLOW}⚠ Mixed signal (check amplitude or frequency){self.RESET}"
            else:
                analysis['Overall'] = f"{self.BLUE}→ REST-like pattern{self.RESET}"
        
        return analysis

