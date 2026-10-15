"""
Persistence-based seizure detection.

Instead of looking for rhythmic patterns, detects sustained periods
of high confidence from the ONNX model. Based on empirical analysis
showing seizures maintain >0.9 confidence for 99.1% of duration.
"""

from collections import deque
from typing import Optional, Dict
import time


class PersistenceDetector:
    """
    Detects seizures based on sustained high confidence from pose model.
    
    Key insight from data analysis:
    - Seizures: 99.1% of time with confidence > 0.9
    - Normal: 4.9% of time with confidence > 0.9
    - Effect size: 13.69 (massive statistical difference)
    
    Strategy: Alert when high percentage of recent predictions exceed threshold.
    """
    
    def __init__(
        self,
        window_seconds: float = 30.0,
        confidence_threshold: float = 0.7,
        persistence_threshold: float = 0.80,  # 80% of window must be high confidence
        min_duration: float = 15.0,  # Must persist for at least 15 seconds
    ):
        """
        Initialize persistence detector.
        
        Args:
            window_seconds: Time window to analyze (default: 30s)
            confidence_threshold: Minimum confidence to count as "high" (default: 0.7)
            persistence_threshold: Fraction of window that must exceed confidence_threshold (default: 0.80)
            min_duration: Minimum duration in seconds before alerting (default: 15s)
        """
        self.window_seconds = window_seconds
        self.confidence_threshold = confidence_threshold
        self.persistence_threshold = persistence_threshold
        self.min_duration = min_duration
        
        # Buffer to store (timestamp, confidence) tuples
        self.predictions = deque()
        
        # State tracking
        self.seizure_start_time: Optional[float] = None
        self.last_alert_time: Optional[float] = None
        
    def add_prediction(self, confidence: float, timestamp: Optional[float] = None) -> Dict:
        """
        Add a new pose model prediction and check for seizure.
        
        Args:
            confidence: Model confidence (0.0 to 1.0)
            timestamp: Time of prediction (default: current time)
            
        Returns:
            Dictionary with detection results:
            {
                'seizure_detected': bool,
                'confidence': float,
                'persistence_score': float,  # Fraction of window above threshold
                'duration': float,  # How long high confidence has persisted
                'window_size': int  # Number of predictions in window
            }
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add new prediction
        self.predictions.append((timestamp, confidence))
        
        # Remove predictions outside window
        cutoff_time = timestamp - self.window_seconds
        while self.predictions and self.predictions[0][0] < cutoff_time:
            self.predictions.popleft()
        
        # Calculate persistence score (fraction above threshold)
        if len(self.predictions) == 0:
            return {
                'seizure_detected': False,
                'confidence': confidence,
                'persistence_score': 0.0,
                'duration': 0.0,
                'window_size': 0
            }
        
        high_confidence_count = sum(1 for _, conf in self.predictions if conf >= self.confidence_threshold)
        persistence_score = high_confidence_count / len(self.predictions)
        
        # Check if we're in a high-persistence state
        is_high_persistence = persistence_score >= self.persistence_threshold
        
        # Track seizure duration
        if is_high_persistence:
            if self.seizure_start_time is None:
                self.seizure_start_time = timestamp
            duration = timestamp - self.seizure_start_time
        else:
            # Reset if persistence drops
            self.seizure_start_time = None
            duration = 0.0
        
        # Determine if we should alert
        seizure_detected = (
            is_high_persistence and 
            duration >= self.min_duration
        )
        
        return {
            'seizure_detected': seizure_detected,
            'confidence': confidence,
            'persistence_score': persistence_score,
            'duration': duration,
            'window_size': len(self.predictions),
            'high_confidence_count': high_confidence_count,
            'seizure_ongoing': is_high_persistence
        }
    
    def reset(self):
        """Clear all state and predictions."""
        self.predictions.clear()
        self.seizure_start_time = None
        self.last_alert_time = None
    
    def get_stats(self) -> Dict:
        """Get current detector statistics."""
        if len(self.predictions) == 0:
            return {
                'window_size': 0,
                'persistence_score': 0.0,
                'mean_confidence': 0.0,
                'max_confidence': 0.0,
                'seizure_duration': 0.0
            }
        
        confidences = [conf for _, conf in self.predictions]
        high_count = sum(1 for conf in confidences if conf >= self.confidence_threshold)
        
        duration = 0.0
        if self.seizure_start_time is not None:
            duration = time.time() - self.seizure_start_time
        
        return {
            'window_size': len(self.predictions),
            'persistence_score': high_count / len(self.predictions),
            'mean_confidence': sum(confidences) / len(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'seizure_duration': duration
        }
    
    def get_state(self) -> Dict:
        """Get current internal state for diagnostics."""
        high_count = sum(1 for _, conf in self.predictions if conf >= self.confidence_threshold)
        persistence_score = high_count / len(self.predictions) if len(self.predictions) > 0 else 0.0
        
        duration = 0.0
        if self.seizure_start_time is not None:
            duration = time.time() - self.seizure_start_time
        
        is_high_persistence = persistence_score >= self.persistence_threshold
        seizure_detected = is_high_persistence and duration >= self.min_duration
        
        return {
            'predictions': list(self.predictions),  # [(timestamp, confidence), ...]
            'persistence_score': persistence_score,
            'duration': duration,
            'seizure_detected': seizure_detected,
            'high_confidence_count': high_count,
            'total_predictions': len(self.predictions)
        }

