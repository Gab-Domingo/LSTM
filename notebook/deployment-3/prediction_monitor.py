"""
Real-time prediction monitoring and visualization.
Provides terminal-based dashboard for monitoring inference accuracy and trends.
"""

import time
import numpy as np
from collections import deque
from typing import List, Dict, Optional
from datetime import datetime


class PredictionMonitor:
    """
    Real-time monitor for model predictions.
    Tracks trends, stability, and provides visual feedback.
    """
    
    def __init__(self, history_size: int = 50):
        """
        Args:
            history_size: Number of predictions to keep in history
        """
        self.history_size = history_size
        self.predictions_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
        # Statistics
        self.total_predictions = 0
        self.prediction_changes = 0
        self.last_prediction = None
        
        # Color codes
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.CYAN = '\033[96m'
        self.MAGENTA = '\033[95m'
        self.BOLD = '\033[1m'
        self.RESET = '\033[0m'
    
    def add_prediction(self, pred_class: str, confidence: float, probabilities: np.ndarray):
        """
        Add a new prediction to the monitor.
        
        Args:
            pred_class: Predicted class name
            confidence: Confidence score (0-1)
            probabilities: Array of probabilities [REST_prob, FIST_prob]
        """
        timestamp = time.time()
        
        # Track prediction changes
        if self.last_prediction is not None and self.last_prediction != pred_class:
            self.prediction_changes += 1
        
        self.last_prediction = pred_class
        
        # Store prediction
        self.predictions_history.append({
            'class': pred_class,
            'confidence': confidence,
            'probabilities': probabilities.copy(),
            'timestamp': timestamp
        })
        self.confidence_history.append(confidence)
        self.timestamps.append(timestamp)
        
        self.total_predictions += 1
    
    def get_stability_score(self, window: int = 10) -> float:
        """
        Calculate prediction stability score (0-1).
        Higher score = more stable predictions.
        
        Args:
            window: Number of recent predictions to analyze
            
        Returns:
            Stability score (0-1)
        """
        if len(self.predictions_history) < 2:
            return 0.0
        
        recent = list(self.predictions_history)[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Count changes in recent window
        changes = sum(1 for i in range(1, len(recent)) 
                     if recent[i]['class'] != recent[i-1]['class'])
        
        # Stability = 1 - (changes / (window - 1))
        stability = 1.0 - (changes / (len(recent) - 1))
        return max(0.0, min(1.0, stability))
    
    def get_confidence_trend(self, window: int = 10) -> str:
        """
        Get confidence trend indicator.
        
        Args:
            window: Number of recent predictions to analyze
            
        Returns:
            Trend string: "↑", "↓", "→", or "?"
        """
        if len(self.confidence_history) < 2:
            return "?"
        
        recent = list(self.confidence_history)[-window:]
        if len(recent) < 2:
            return "?"
        
        # Simple linear trend
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        diff = second_half - first_half
        threshold = 0.05  # 5% change threshold
        
        if diff > threshold:
            return "↑"
        elif diff < -threshold:
            return "↓"
        else:
            return "→"
    
    def get_prediction_distribution(self, window: int = 20) -> Dict[str, int]:
        """
        Get distribution of predictions in recent window.
        
        Args:
            window: Number of recent predictions to analyze
            
        Returns:
            Dictionary with class counts
        """
        if len(self.predictions_history) == 0:
            return {'REST': 0, 'FIST': 0}
        
        recent = list(self.predictions_history)[-window:]
        distribution = {'REST': 0, 'FIST': 0}
        
        for pred in recent:
            distribution[pred['class']] = distribution.get(pred['class'], 0) + 1
        
        return distribution
    
    def get_quality_score(self) -> float:
        """
        Calculate overall prediction quality score (0-1).
        Based on confidence and stability.
        
        Returns:
            Quality score (0-1)
        """
        if len(self.predictions_history) == 0:
            return 0.0
        
        # Average confidence
        avg_confidence = np.mean(list(self.confidence_history))
        
        # Stability
        stability = self.get_stability_score()
        
        # Quality = weighted combination
        # 70% confidence, 30% stability
        quality = 0.7 * avg_confidence + 0.3 * stability
        
        return max(0.0, min(1.0, quality))
    
    def render_timeline(self, width: int = 50) -> str:
        """
        Render ASCII timeline of recent predictions.
        
        Args:
            width: Width of timeline in characters
            
        Returns:
            Timeline string
        """
        if len(self.predictions_history) == 0:
            return " " * width + " (no predictions yet)"
        
        # Get recent predictions
        recent = list(self.predictions_history)[-width:]
        
        timeline = []
        for pred in recent:
            if pred['class'] == 'FIST':
                timeline.append(self.GREEN + '█' + self.RESET)
            else:
                timeline.append(self.BLUE + '█' + self.RESET)
        
        # Pad if needed
        while len(timeline) < width:
            timeline.insert(0, ' ')
        
        return ''.join(timeline[-width:])
    
    def render_confidence_bars(self, width: int = 30) -> str:
        """
        Render confidence bars for recent predictions.
        
        Args:
            width: Width of bars in characters
            
        Returns:
            Confidence bars string
        """
        if len(self.confidence_history) == 0:
            return " " * width + " (no data)"
        
        recent = list(self.confidence_history)[-width:]
        
        bars = []
        for conf in recent:
            bar_length = int(conf * width)
            bar = '█' * bar_length + '░' * (width - bar_length)
            
            # Color based on confidence
            if conf >= 0.8:
                color = self.GREEN
            elif conf >= 0.6:
                color = self.YELLOW
            else:
                color = self.RED
            
            bars.append(color + bar[0] + self.RESET)
        
        return ''.join(bars[-width:])
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with statistics
        """
        if len(self.predictions_history) == 0:
            return {
                'total': 0,
                'avg_confidence': 0.0,
                'stability': 0.0,
                'quality': 0.0,
                'rest_count': 0,
                'fist_count': 0,
                'change_rate': 0.0
            }
        
        recent = list(self.predictions_history)
        rest_count = sum(1 for p in recent if p['class'] == 'REST')
        fist_count = sum(1 for p in recent if p['class'] == 'FIST')
        
        avg_confidence = np.mean(list(self.confidence_history))
        stability = self.get_stability_score()
        quality = self.get_quality_score()
        
        # Change rate (changes per prediction)
        change_rate = (self.prediction_changes / max(1, self.total_predictions - 1))
        
        return {
            'total': len(recent),
            'avg_confidence': avg_confidence,
            'stability': stability,
            'quality': quality,
            'rest_count': rest_count,
            'fist_count': fist_count,
            'change_rate': change_rate
        }
    
    def print_dashboard(self, clear_screen: bool = False):
        """
        Print real-time monitoring dashboard.
        
        Args:
            clear_screen: Whether to clear screen before printing
        """
        if clear_screen:
            print("\033[2J\033[H", end='')  # Clear screen and move cursor to top
        
        stats = self.get_summary_stats()
        
        # Compact header
        print("\n" + "─"*70)
        print(f"{self.BOLD}📊 Real-Time Prediction Monitor{self.RESET}")
        print("─"*70)
        
        # Show waiting message if no predictions yet
        if stats['total'] == 0:
            print(f"\n   {self.YELLOW}⏳ Waiting for predictions...{self.RESET}")
            print("─"*70)
            return
        
        # Current prediction status (most recent)
        if len(self.predictions_history) > 0:
            latest = self.predictions_history[-1]
            symbol = '✊' if latest['class'] == 'FIST' else '🛑'
            class_color = self.GREEN if latest['class'] == 'FIST' else self.BLUE
            conf_color = self.GREEN if latest['confidence'] >= 0.8 else (
                self.YELLOW if latest['confidence'] >= 0.6 else self.RED
            )
            
            print(f"\n{self.BOLD}Current Prediction:{self.RESET}")
            print(f"   {class_color}{symbol} {latest['class']}{self.RESET}  "
                  f"Confidence: {conf_color}{latest['confidence']:.1%}{self.RESET}  "
                  f"| REST: {latest['probabilities'][0]:.1%}  FIST: {latest['probabilities'][1]:.1%}")
        
        # Timeline visualization (compact)
        print(f"\n{self.BOLD}Recent Timeline:{self.RESET} (most recent →)")
        timeline = self.render_timeline(width=60)
        print(f"   {timeline}")
        print(f"   {self.BLUE}█{self.RESET} = REST  {self.GREEN}█{self.RESET} = FIST")
        
        # Quick stats in one line
        print(f"\n{self.BOLD}Stats:{self.RESET} "
              f"Total: {stats['total']} | "
              f"Avg Confidence: {self._color_confidence(stats['avg_confidence'])} | "
              f"Stability: {self._color_stability(stats['stability'])} | "
              f"{self.BLUE}REST:{self.RESET} {stats['rest_count']}  "
              f"{self.GREEN}FIST:{self.RESET} {stats['fist_count']}")
        
        # Overall quality indicator (compact)
        quality = stats['quality']
        if quality >= 0.8:
            quality_color = self.GREEN
            quality_label = "EXCELLENT"
        elif quality >= 0.6:
            quality_color = self.YELLOW
            quality_label = "GOOD"
        else:
            quality_color = self.RED
            quality_label = "NEEDS ATTENTION"
        
        print(f"   Quality: {quality_color}{quality_label}{self.RESET} ({quality:.1%})")
        print("─"*70)
    
    def _color_confidence(self, conf: float) -> str:
        """Color code confidence value."""
        if conf >= 0.8:
            return f"{self.GREEN}{conf:.1%}{self.RESET}"
        elif conf >= 0.6:
            return f"{self.YELLOW}{conf:.1%}{self.RESET}"
        else:
            return f"{self.RED}{conf:.1%}{self.RESET}"
    
    def _color_stability(self, stability: float) -> str:
        """Color code stability value."""
        if stability >= 0.8:
            return f"{self.GREEN}{stability:.1%}{self.RESET}"
        elif stability >= 0.6:
            return f"{self.YELLOW}{stability:.1%}{self.RESET}"
        else:
            return f"{self.RED}{stability:.1%}{self.RESET}"

