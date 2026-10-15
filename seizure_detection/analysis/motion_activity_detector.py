"""
Motion-based activity detector for Stage 2.

Analyzes motion patterns to detect seizure-like activity WITHOUT requiring pose landmarks.
This is robust to occlusion, lighting issues, and pose detection failures.

Uses optical flow and temporal patterns to distinguish:
- Normal movements (rolling over, adjusting position)
- Suspicious activity (rapid, rhythmic, or violent movements)
"""
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from ..config import SeizureDetectionConfig
from ..utils.logger import setup_logger
from .motion_features import MotionFeatureExtractor

logger = setup_logger(__name__)


class MotionActivityDetector:
    """
    Detects suspicious activity patterns using motion analysis only.
    
    Detection strategies:
    - Rhythmic motion (repetitive patterns)
    - Sudden acceleration (violent movements)
    - Sustained high motion (prolonged activity)
    - Abnormal motion patterns (irregular, chaotic)
    """
    
    def __init__(self, config: SeizureDetectionConfig = SeizureDetectionConfig()):
        self.config = config
        self.motion_extractor = MotionFeatureExtractor(history_length=30)  # 6 seconds at 5 FPS
        
        # Detection thresholds
        self.rhythmic_frequency_range = (1.0, 5.0)  # 1-5 Hz (seizure-like rhythms)
        self.high_motion_threshold = 5.0  # Pixels per frame
        self.acceleration_threshold = 10.0  # Change in motion magnitude
        
        # History tracking
        self.detection_score = 0.0
        self.last_detection_time: Optional[datetime] = None
        
    def process_frame(self, frame: np.ndarray, timestamp: datetime) -> Dict:
        """
        Process frame and analyze motion patterns.
        
        Args:
            frame: Video frame (BGR format)
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with detection results:
                - confidence: Overall confidence [0-1]
                - detected: Bool indicating if suspicious activity detected
                - details: Dict with breakdown of detection reasons
        """
        # Extract motion features
        motion_features = self.motion_extractor.process_frame(frame)
        stats = motion_features['stats']
        
        # Initialize result
        result = {
            'confidence': 0.0,
            'detected': False,
            'details': {
                'rhythmic': False,
                'rhythmic_score': 0.0,
                'sudden': False,
                'sudden_score': 0.0,
                'sustained': False,
                'sustained_score': 0.0,
                'motion_magnitude': stats.get('avg_magnitude', 0.0),
                'motion_density': stats.get('motion_density', 0.0),
            }
        }
        
        # Need sufficient history for analysis
        if len(self.motion_extractor.motion_magnitude_history) < 10:
            return result
        
        # Detection 1: Rhythmic motion (common in seizures)
        rhythmic_detected, rhythmic_score = self._detect_rhythmic_motion()
        result['details']['rhythmic'] = rhythmic_detected
        result['details']['rhythmic_score'] = rhythmic_score
        
        # Detection 2: Sudden acceleration (violent movements)
        sudden_detected, sudden_score = self._detect_sudden_acceleration()
        result['details']['sudden'] = sudden_detected
        result['details']['sudden_score'] = sudden_score
        
        # Detection 3: Sustained high motion (prolonged activity)
        sustained_detected, sustained_score = self._detect_sustained_motion()
        result['details']['sustained'] = sustained_detected
        result['details']['sustained_score'] = sustained_score
        
        # Combine detection scores (weighted average)
        scores = []
        if rhythmic_detected:
            scores.append(rhythmic_score * 0.4)  # Rhythmic is strong indicator
        if sudden_detected:
            scores.append(sudden_score * 0.3)    # Sudden movement significant
        if sustained_detected:
            scores.append(sustained_score * 0.3)  # Sustained motion relevant
        
        if scores:
            result['confidence'] = min(1.0, sum(scores))
            result['detected'] = result['confidence'] >= self.config.CONFIDENCE_THRESHOLD
            
            if result['detected']:
                self.last_detection_time = timestamp
                logger.debug(f"Suspicious activity detected: confidence={result['confidence']:.2f}, "
                           f"rhythmic={rhythmic_detected}, sudden={sudden_detected}, sustained={sustained_detected}")
        
        return result
    
    def _detect_rhythmic_motion(self) -> tuple[bool, float]:
        """
        Detect rhythmic patterns in motion (common in seizures).
        
        Returns:
            Tuple of (detected, confidence)
        """
        magnitudes = list(self.motion_extractor.motion_magnitude_history)
        if len(magnitudes) < 20:
            return False, 0.0
        
        try:
            # Analyze recent motion (last 4 seconds = 20 frames at 5 FPS)
            recent = magnitudes[-20:]
            mean_mag = np.mean(recent)
            
            # Need sufficient motion to analyze rhythm
            if mean_mag < 2.0:
                return False, 0.0
            
            # Count zero crossings (crossings of mean line)
            crossings = 0
            for i in range(1, len(recent)):
                if (recent[i-1] - mean_mag) * (recent[i] - mean_mag) < 0:
                    crossings += 1
            
            # Estimate frequency from crossings
            duration = 4.0  # seconds
            frequency = crossings / (2 * duration)  # Full cycles
            
            # Check if in seizure frequency range (1-5 Hz)
            min_freq, max_freq = self.rhythmic_frequency_range
            if min_freq <= frequency <= max_freq:
                # Higher confidence for stronger rhythmic patterns
                std_dev = np.std(recent)
                regularity = 1.0 - min(1.0, std_dev / (mean_mag + 1e-6))
                confidence = min(1.0, frequency / 3.0) * regularity
                
                return True, confidence
            
        except Exception as e:
            logger.warning(f"Error in rhythmic motion detection: {e}")
        
        return False, 0.0
    
    def _detect_sudden_acceleration(self) -> tuple[bool, float]:
        """
        Detect sudden, violent accelerations in motion.
        
        Returns:
            Tuple of (detected, confidence)
        """
        magnitudes = list(self.motion_extractor.motion_magnitude_history)
        if len(magnitudes) < 5:
            return False, 0.0
        
        try:
            # Check recent motion changes (last 1 second = 5 frames)
            recent = magnitudes[-5:]
            
            # Calculate frame-to-frame changes (acceleration proxy)
            accelerations = []
            for i in range(1, len(recent)):
                accel = abs(recent[i] - recent[i-1])
                accelerations.append(accel)
            
            max_accel = max(accelerations)
            
            # Detect sudden spike in motion
            if max_accel > self.acceleration_threshold:
                # Confidence based on magnitude of acceleration
                confidence = min(1.0, max_accel / 20.0)
                return True, confidence
            
        except Exception as e:
            logger.warning(f"Error in sudden acceleration detection: {e}")
        
        return False, 0.0
    
    def _detect_sustained_motion(self) -> tuple[bool, float]:
        """
        Detect sustained high-intensity motion (prolonged activity).
        
        Returns:
            Tuple of (detected, confidence)
        """
        magnitudes = list(self.motion_extractor.motion_magnitude_history)
        densities = list(self.motion_extractor.motion_density_history)
        
        if len(magnitudes) < 15:
            return False, 0.0
        
        try:
            # Check last 3 seconds (15 frames at 5 FPS)
            recent_mag = magnitudes[-15:]
            recent_dens = densities[-15:]
            
            avg_magnitude = np.mean(recent_mag)
            avg_density = np.mean(recent_dens)
            
            # High sustained motion: both magnitude and density elevated
            if avg_magnitude > self.high_motion_threshold and avg_density > 0.15:
                # Confidence based on how sustained it is
                consistency = 1.0 - (np.std(recent_mag) / (avg_magnitude + 1e-6))
                confidence = min(1.0, (avg_magnitude / 10.0) * consistency)
                
                return True, confidence
            
        except Exception as e:
            logger.warning(f"Error in sustained motion detection: {e}")
        
        return False, 0.0
    
    def reset(self):
        """Clear all history."""
        self.motion_extractor.reset()
        self.detection_score = 0.0
        self.last_detection_time = None
        logger.info("Motion activity detector reset")
