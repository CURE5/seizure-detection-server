"""
Seizure detector module - analyzes motion patterns to detect potential seizures.
"""
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from ..config import SeizureDetectionConfig
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SeizureDetector:
    """
    Detects potential seizure activity based on pose landmark analysis.
    
    Uses multiple detection strategies:
    - Velocity/acceleration analysis
    - Rhythmic motion detection
    - Sudden movement detection
    """
    
    def __init__(self, config: SeizureDetectionConfig = SeizureDetectionConfig()):
        self.config = config
        self.pose_history = deque(maxlen=config.HISTORY_LENGTH)
        self.velocity_history = deque(maxlen=config.HISTORY_LENGTH)
        self.timestamps = deque(maxlen=config.HISTORY_LENGTH)
        
        self.detection_score = 0.0
        self.last_detection_time: Optional[datetime] = None
        
        # Track specific body parts for analysis
        self.tracked_landmarks = [
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_HIP', 'RIGHT_HIP'
        ]
        
    def add_pose_data(self, landmarks: Optional[List[Dict]], timestamp: datetime):
        """
        Add pose data to history for analysis.
        
        Args:
            landmarks: List of detected pose landmarks
            timestamp: Timestamp of the frame
        """
        if not landmarks or len(landmarks) == 0:
            return
        
        # Extract key landmarks
        key_points = self._extract_key_landmarks(landmarks)
        
        if key_points:
            self.pose_history.append(key_points)
            self.timestamps.append(timestamp)
            
            # Calculate velocities if we have history
            if len(self.pose_history) >= 2:
                velocity = self._calculate_velocity(
                    self.pose_history[-2],
                    self.pose_history[-1],
                    (self.timestamps[-1] - self.timestamps[-2]).total_seconds()
                )
                self.velocity_history.append(velocity)
    
    def _extract_key_landmarks(self, landmarks: List[Dict]) -> Optional[Dict]:
        """Extract and flatten key landmarks for analysis."""
        if not landmarks:
            return None
        
        # Use first detected person
        person = landmarks[0]
        key_points = {}
        
        for key in self.tracked_landmarks:
            if key in person:
                key_points[key] = np.array([
                    person[key].get('x', 0),
                    person[key].get('y', 0),
                    person[key].get('z', 0)
                ])
        
        return key_points if key_points else None
    
    def _calculate_velocity(self, prev_points: Dict, curr_points: Dict, dt: float) -> Dict:
        """Calculate velocity for each landmark."""
        if dt <= 0:
            return {}
        
        velocities = {}
        for key in prev_points:
            if key in curr_points:
                velocities[key] = (curr_points[key] - prev_points[key]) / dt
        
        return velocities
    
    def _calculate_acceleration(self) -> Optional[float]:
        """Calculate average acceleration magnitude."""
        if len(self.velocity_history) < 2:
            return None
        
        accelerations = []
        for i in range(1, len(self.velocity_history)):
            dt = (self.timestamps[i] - self.timestamps[i-1]).total_seconds()
            if dt <= 0:
                continue
            
            for key in self.velocity_history[i]:
                if key in self.velocity_history[i-1]:
                    accel = (self.velocity_history[i][key] - self.velocity_history[i-1][key]) / dt
                    accelerations.append(np.linalg.norm(accel))
        
        return np.mean(accelerations) if accelerations else None
    
    def _detect_rhythmic_motion(self) -> Tuple[bool, float]:
        """
        Detect rhythmic patterns in motion (common in seizures).
        
        Returns:
            Tuple of (detected, confidence)
        """
        if len(self.velocity_history) < 10:
            return False, 0.0
        
        # Analyze frequency content of motion
        try:
            # Get velocity magnitudes over time
            magnitudes = []
            for vel_dict in self.velocity_history:
                mags = [np.linalg.norm(v) for v in vel_dict.values()]
                magnitudes.append(np.mean(mags) if mags else 0.0)
            
            # Simple frequency detection using zero crossings
            mean_mag = np.mean(magnitudes)
            crossings = 0
            for i in range(1, len(magnitudes)):
                if (magnitudes[i-1] - mean_mag) * (magnitudes[i] - mean_mag) < 0:
                    crossings += 1
            
            # Estimate frequency
            duration = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            if duration > 0:
                frequency = crossings / (2 * duration)
                
                # Check if frequency is in seizure range
                min_freq, max_freq = self.config.RHYTHMIC_FREQUENCY_RANGE
                if min_freq <= frequency <= max_freq:
                    confidence = min(1.0, (np.std(magnitudes) / (mean_mag + 1e-6)))
                    return True, confidence
            
        except Exception as e:
            logger.warning(f"Error in rhythmic motion detection: {e}")
        
        return False, 0.0
    
    def _detect_sudden_movement(self) -> Tuple[bool, float]:
        """
        Detect sudden, sharp movements.
        
        Returns:
            Tuple of (detected, confidence)
        """
        if len(self.velocity_history) < 3:
            return False, 0.0
        
        # Check recent velocities
        recent_velocities = list(self.velocity_history)[-5:]
        
        max_velocity = 0.0
        for vel_dict in recent_velocities:
            for vel in vel_dict.values():
                mag = np.linalg.norm(vel)
                max_velocity = max(max_velocity, mag)
        
        if max_velocity > self.config.VELOCITY_THRESHOLD:
            confidence = min(1.0, max_velocity / (self.config.VELOCITY_THRESHOLD * 2))
            return True, confidence
        
        return False, 0.0
    
    def _detect_high_acceleration(self) -> Tuple[bool, float]:
        """
        Detect high acceleration movements.
        
        Returns:
            Tuple of (detected, confidence)
        """
        accel = self._calculate_acceleration()
        if accel is None:
            return False, 0.0
        
        if accel > self.config.ACCELERATION_THRESHOLD:
            confidence = min(1.0, accel / (self.config.ACCELERATION_THRESHOLD * 2))
            return True, confidence
        
        return False, 0.0
    
    def analyze(self) -> Dict:
        """
        Analyze current motion data for seizure indicators.
        
        Returns:
            Dictionary with detection results and confidence scores
        """
        if len(self.pose_history) < 5:
            return {
                'detected': False,
                'confidence': 0.0,
                'indicators': {}
            }
        
        # Run all detection methods
        rhythmic_detected, rhythmic_conf = self._detect_rhythmic_motion()
        sudden_detected, sudden_conf = self._detect_sudden_movement()
        accel_detected, accel_conf = self._detect_high_acceleration()
        
        # Combine scores with weights
        indicators = {
            'rhythmic_motion': {
                'detected': rhythmic_detected,
                'confidence': rhythmic_conf
            },
            'sudden_movement': {
                'detected': sudden_detected,
                'confidence': sudden_conf
            },
            'high_acceleration': {
                'detected': accel_detected,
                'confidence': accel_conf
            }
        }
        
        # Weighted average (rhythmic motion is stronger indicator)
        total_confidence = (
            rhythmic_conf * 0.5 +
            sudden_conf * 0.25 +
            accel_conf * 0.25
        )
        
        self.detection_score = total_confidence
        detected = total_confidence >= self.config.CONFIDENCE_THRESHOLD
        
        if detected:
            self.last_detection_time = datetime.now()
            logger.warning(
                f"Seizure detected! Confidence: {total_confidence:.2f}, "
                f"Indicators: rhythmic={rhythmic_conf:.2f}, "
                f"sudden={sudden_conf:.2f}, accel={accel_conf:.2f}"
            )
        
        return {
            'detected': detected,
            'confidence': total_confidence,
            'indicators': indicators,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset(self):
        """Reset the detector state."""
        self.pose_history.clear()
        self.velocity_history.clear()
        self.timestamps.clear()
        self.detection_score = 0.0
        logger.info("Seizure detector reset")
