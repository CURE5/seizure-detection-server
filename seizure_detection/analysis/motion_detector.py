"""
Improved motion detection with adaptive FPS control.
"""
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
from ..config import MOTION_THRESHOLD, MOTION_DEGREE_THRESHOLD, FPSConfig
from ..utils.logger import setup_logger
from ..utils.states import SleepMode

logger = setup_logger(__name__)


class MotionDetector:
    """
    Detects motion in video frames and manages adaptive FPS based on activity.
    """
    
    def __init__(self):
        self.last_frame: Optional[np.ndarray] = None
        self.motion_degree: float = 0.0
        self.motion_last_detected_at: datetime = datetime.now()
        self.sleep_mode: SleepMode = SleepMode.ACTIVE
        
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect motion in the current frame compared to the previous frame.
        
        Args:
            frame: Current video frame (BGR format)
            
        Returns:
            Tuple of (motion_detected, motion_degree)
        """
        # Convert to grayscale for better motion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.last_frame is None:
            self.last_frame = gray_frame
            return False, 0.0
        
        try:
            # Compute the absolute difference between frames
            diff = cv2.absdiff(self.last_frame, gray_frame)
            
            # Apply binary threshold
            _, diff_thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # Calculate motion degree (percentage of changed pixels)
            motion_pixels = np.sum(diff_thresh > 0)
            total_pixels = diff_thresh.size
            motion_degree = motion_pixels / total_pixels
            
            self.motion_degree = motion_degree
            motion_detected = motion_degree > MOTION_DEGREE_THRESHOLD
            
            # Update "last motion" timestamp for ANY motion above a very low threshold
            # This keeps system awake even for small movements (sleeping baby)
            # Use 1/3 of alert threshold for "keep awake" threshold
            keep_awake_threshold = MOTION_DEGREE_THRESHOLD * 0.3
            any_motion = motion_degree > keep_awake_threshold
            
            if any_motion:
                self.motion_last_detected_at = datetime.now()
                if motion_detected and self.sleep_mode != SleepMode.ACTIVE:
                    logger.info(f"Motion detected, waking up. Degree: {motion_degree:.4f}, Alert threshold: {MOTION_DEGREE_THRESHOLD}")
                # Only force to ACTIVE if significant motion detected
                if motion_detected:
                    self.sleep_mode = SleepMode.ACTIVE
            else:
                # Log when there's truly no motion for debugging
                if self.sleep_mode == SleepMode.ACTIVE:
                    time_since = self.get_time_since_last_motion().total_seconds()
                    if time_since > 30:  # Only log after 30 seconds of no motion
                        logger.debug(f"No motion detected. Degree: {motion_degree:.4f}, Time since last: {time_since:.0f}s")
            
            # Update last frame
            self.last_frame = gray_frame
            
            return motion_detected, motion_degree
            
        except Exception as e:
            logger.error(f"Error detecting motion: {e}")
            self.last_frame = gray_frame
            return False, 0.0
    
    def get_time_since_last_motion(self) -> timedelta:
        """Get time elapsed since last motion was detected."""
        return datetime.now() - self.motion_last_detected_at
    
    def update_sleep_mode(self) -> SleepMode:
        """
        Update sleep mode based on time since last motion.
        
        Returns:
            Current sleep mode
        """
        time_since = self.get_time_since_last_motion().total_seconds()
        
        previous_mode = self.sleep_mode
        
        # If motion was detected very recently (within 1 second), stay ACTIVE
        # This prevents the race condition where detect_motion sets ACTIVE
        # but then update_sleep_mode immediately recalculates
        if time_since < 1.0:
            self.sleep_mode = SleepMode.ACTIVE
        elif time_since < FPSConfig.TRANSITION_THRESHOLD:
            self.sleep_mode = SleepMode.ACTIVE
        elif time_since < FPSConfig.LIGHT_THRESHOLD:
            self.sleep_mode = SleepMode.TRANSITIONING
        else:
            # Check motion degree for deeper sleep
            if self.motion_degree < MOTION_DEGREE_THRESHOLD * 0.5:
                self.sleep_mode = SleepMode.DEEP_SLEEP
            else:
                self.sleep_mode = SleepMode.LIGHT_SLEEP
        
        if previous_mode != self.sleep_mode:
            logger.info(
                f"Sleep mode changed: {previous_mode.name} -> {self.sleep_mode.name} "
                f"(time since motion: {time_since:.1f}s, motion_degree: {self.motion_degree:.4f})"
            )
        
        return self.sleep_mode
    
    def get_adaptive_fps(self) -> float:
        """
        Get adaptive FPS based on current sleep mode.
        
        Returns:
            Recommended FPS
        """
        self.update_sleep_mode()
        
        fps_map = {
            SleepMode.ACTIVE: FPSConfig.ACTIVE,
            SleepMode.TRANSITIONING: FPSConfig.SLEEP_TRANSITION,
            SleepMode.LIGHT_SLEEP: FPSConfig.SLEEP_LIGHT,
            SleepMode.DEEP_SLEEP: FPSConfig.SLEEP_DEEP
        }
        
        return fps_map.get(self.sleep_mode, FPSConfig.ACTIVE)
    
    def reset(self):
        """Reset motion detector state."""
        self.last_frame = None
        self.motion_degree = 0.0
        self.motion_last_detected_at = datetime.now()
        self.sleep_mode = SleepMode.ACTIVE
        logger.info("Motion detector reset")
