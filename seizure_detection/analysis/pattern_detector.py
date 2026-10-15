"""
Rhythmic pattern detection for hypermotor seizures.

Analyzes sequences of pose model predictions over time to detect
the characteristic spasm→calm→spasm rhythm of hypermotor seizures.

No retraining needed - uses existing pose predictions to find patterns.
"""
import numpy as np
from typing import List, Dict
from collections import deque
from datetime import datetime
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RhythmicPatternDetector:
    """
    Detects rhythmic spasm patterns characteristic of hypermotor seizures.
    
    Hypermotor seizure characteristics:
    - Spasms: High pose confidence (>0.5) lasting 1-3 seconds
    - Calm periods: Low pose confidence (<0.3) lasting 4-20 seconds
    - Pattern: Spasm → Calm → Spasm (repeats 3+ times)
    - Duration: Typically 30 seconds to several minutes
    """
    
    def __init__(
        self,
        history_seconds: int = 60,
        fps: float = 5.0,
        spasm_threshold: float = 0.5,
        calm_threshold: float = 0.3,
        min_spasms: int = 3,
        min_interval: float = 4.0,
        max_interval: float = 20.0
    ):
        """
        Initialize rhythmic pattern detector.
        
        Args:
            history_seconds: How many seconds of predictions to analyze
            fps: Frames per second for timing calculations
            spasm_threshold: Confidence above this = spasm
            calm_threshold: Confidence below this = calm
            min_spasms: Minimum number of spasms to confirm pattern
            min_interval: Minimum seconds between spasms (too fast = not hypermotor)
            max_interval: Maximum seconds between spasms (too slow = separate events)
        """
        self.history_length = int(history_seconds * fps)
        self.fps = fps
        self.spasm_threshold = spasm_threshold
        self.calm_threshold = calm_threshold
        self.min_spasms = min_spasms
        self.min_interval = min_interval
        self.max_interval = max_interval
        
        # Prediction history
        self.predictions = deque(maxlen=self.history_length)
        self.timestamps = deque(maxlen=self.history_length)
        
        # Pattern state
        self.last_pattern_check = None
        self.detected_spasms: List[Dict] = []  # Recent spasm events
        
    def add_prediction(self, confidence: float, timestamp: datetime):
        """Add a pose model prediction to history."""
        self.predictions.append(confidence)
        self.timestamps.append(timestamp)
    
    def detect_pattern(self) -> Dict:
        """
        Analyze prediction history for rhythmic spasm pattern.
        
        Returns:
            Dictionary with detection result:
                - pattern_detected: bool
                - confidence: float (0-1)
                - spasm_count: int
                - avg_interval: float (seconds)
                - intervals: List[float]
                - rhythm_score: float (consistency of intervals)
        """
        if len(self.predictions) < self.min_spasms * 10:  # Need enough data
            return self._empty_result()
        
        # Find spasm peaks in prediction history
        spasms = self._find_spasms()
        
        if len(spasms) < self.min_spasms:
            return self._empty_result()
        
        # Calculate intervals between spasms
        intervals = []
        for i in range(1, len(spasms)):
            interval = (spasms[i]['timestamp'] - spasms[i-1]['timestamp']).total_seconds()
            intervals.append(interval)
        
        # Check if intervals are within expected range (4-20s for hypermotor)
        valid_intervals = [
            i for i in intervals 
            if self.min_interval <= i <= self.max_interval
        ]
        
        if len(valid_intervals) < (self.min_spasms - 1):
            # Not enough valid spasm intervals
            return self._empty_result()
        
        # Calculate rhythm consistency
        avg_interval = np.mean(valid_intervals)
        std_interval = np.std(valid_intervals)
        rhythm_score = 1.0 - min(std_interval / avg_interval, 1.0) if avg_interval > 0 else 0.0
        
        # Calculate overall confidence
        # Based on: number of spasms, rhythm consistency, spasm confidence
        spasm_confidences = [s['confidence'] for s in spasms]
        avg_spasm_confidence = np.mean(spasm_confidences)
        
        # Weighted score
        pattern_confidence = (
            0.4 * min(len(spasms) / 5, 1.0) +  # More spasms = higher confidence
            0.3 * rhythm_score +                # Consistent rhythm = higher confidence
            0.3 * avg_spasm_confidence          # Strong spasms = higher confidence
        )
        
        pattern_detected = (
            len(spasms) >= self.min_spasms and
            rhythm_score > 0.5 and  # Intervals should be fairly consistent
            avg_spasm_confidence > self.spasm_threshold
        )
        
        logger.debug(
            f"Pattern analysis: {len(spasms)} spasms, "
            f"avg interval={avg_interval:.1f}s, "
            f"rhythm={rhythm_score:.2f}, "
            f"confidence={pattern_confidence:.2f}"
        )
        
        return {
            'pattern_detected': pattern_detected,
            'confidence': pattern_confidence,
            'spasm_count': len(spasms),
            'avg_interval': avg_interval,
            'intervals': valid_intervals,
            'rhythm_score': rhythm_score,
            'spasms': spasms
        }
    
    def _find_spasms(self) -> List[Dict]:
        """
        Find spasm events in prediction history.
        
        A spasm is:
        - Sustained high confidence (>threshold) for 1-3 seconds
        - Preceded and followed by lower confidence
        
        Returns:
            List of spasm events with timestamp, duration, peak confidence
        """
        if len(self.predictions) == 0:
            return []
        
        spasms = []
        predictions = list(self.predictions)
        timestamps = list(self.timestamps)
        
        in_spasm = False
        spasm_start_idx = 0
        spasm_confidences = []
        
        for i, conf in enumerate(predictions):
            if conf >= self.spasm_threshold:
                if not in_spasm:
                    # Spasm starting
                    in_spasm = True
                    spasm_start_idx = i
                    spasm_confidences = [conf]
                else:
                    # Spasm continuing
                    spasm_confidences.append(conf)
            else:
                if in_spasm:
                    # Spasm ending
                    in_spasm = False
                    
                    # Calculate spasm duration
                    duration = len(spasm_confidences) / self.fps
                    
                    # Only count if duration is reasonable (0.2-5 seconds)
                    if 0.2 <= duration <= 5.0:
                        spasms.append({
                            'timestamp': timestamps[spasm_start_idx],
                            'duration': duration,
                            'confidence': max(spasm_confidences),
                            'avg_confidence': np.mean(spasm_confidences),
                            'start_idx': spasm_start_idx,
                            'end_idx': i - 1
                        })
                    
                    spasm_confidences = []
        
        # Handle case where spasm is ongoing at end of buffer
        if in_spasm and len(spasm_confidences) > 0:
            duration = len(spasm_confidences) / self.fps
            if 0.2 <= duration <= 5.0:
                spasms.append({
                    'timestamp': timestamps[spasm_start_idx],
                    'duration': duration,
                    'confidence': max(spasm_confidences),
                    'avg_confidence': np.mean(spasm_confidences),
                    'start_idx': spasm_start_idx,
                    'end_idx': len(predictions) - 1
                })
        
        return spasms
    
    def _empty_result(self) -> Dict:
        """Return empty result when no pattern detected."""
        return {
            'pattern_detected': False,
            'confidence': 0.0,
            'spasm_count': 0,
            'avg_interval': 0.0,
            'intervals': [],
            'rhythm_score': 0.0,
            'spasms': []
        }
    
    def get_stats(self) -> Dict:
        """Get current detector statistics."""
        return {
            'history_size': len(self.predictions),
            'history_seconds': len(self.predictions) / self.fps if self.fps > 0 else 0,
            'spasm_threshold': self.spasm_threshold,
            'calm_threshold': self.calm_threshold,
            'min_spasms': self.min_spasms
        }
