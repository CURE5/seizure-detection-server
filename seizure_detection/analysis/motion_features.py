"""
Motion-based feature extraction for seizure detection.

This module provides portable, robust motion analysis that works regardless of:
- Body pose/orientation
- Occlusion (blankets, etc.)
- Lighting conditions
- Person detection quality

Designed for fixed camera monitoring of a mostly-stationary subject.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import deque
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MotionFeatureExtractor:
    """
    Extracts motion-based features using optical flow and temporal analysis.
    
    Features extracted:
    - Optical flow magnitude (how much motion)
    - Motion direction (where motion is going)
    - Motion density (percentage of frame moving)
    - Temporal patterns (rhythm, acceleration)
    """
    
    def __init__(self, 
                 history_length: int = 30,  # 6 seconds at 5 FPS
                 roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize motion feature extractor.
        
        Args:
            history_length: Number of frames to keep in history (default: 30 = 6 seconds at 5 FPS)
            roi: Region of interest (x, y, w, h). If None, uses full frame.
        """
        self.history_length = history_length
        self.roi = roi
        
        # Frame history
        self.frame_history = deque(maxlen=history_length)
        self.gray_history = deque(maxlen=2)  # Only need current and previous for optical flow
        
        # Motion feature history
        self.motion_magnitude_history = deque(maxlen=history_length)
        self.motion_density_history = deque(maxlen=history_length)
        self.motion_direction_history = deque(maxlen=history_length)
        
        # Optical flow parameters (Farneback)
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process a frame and extract motion features.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary containing motion features:
                - flow_magnitude: 2D array of motion intensity
                - flow_direction: 2D array of motion direction (radians)
                - motion_mask: Binary mask of moving regions
                - stats: Dictionary of scalar statistics
        """
        # Convert to grayscale and apply ROI if specified
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.roi:
            x, y, w, h = self.roi
            gray = gray[y:y+h, x:x+w]
        
        self.gray_history.append(gray)
        self.frame_history.append(frame)
        
        # Need at least 2 frames for optical flow
        if len(self.gray_history) < 2:
            return self._empty_features(gray.shape)
        
        # Calculate optical flow
        prev_gray = self.gray_history[-2]
        curr_gray = self.gray_history[-1]
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self.flow_params
        )
        
        # Extract flow components
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        direction = np.arctan2(flow_y, flow_x)  # Radians: -π to π
        
        # Create motion mask (threshold for "significant" motion)
        motion_threshold = 0.5  # pixels per frame at 5 FPS
        motion_mask = (magnitude > motion_threshold).astype(np.uint8)
        
        # Calculate statistics
        avg_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        motion_density = np.sum(motion_mask) / motion_mask.size  # Percentage moving
        
        # Store in history
        self.motion_magnitude_history.append(avg_magnitude)
        self.motion_density_history.append(motion_density)
        self.motion_direction_history.append(np.mean(direction[motion_mask > 0]) if np.any(motion_mask) else 0.0)
        
        stats = {
            'avg_magnitude': avg_magnitude,
            'max_magnitude': max_magnitude,
            'motion_density': motion_density,
            'motion_area': np.sum(motion_mask),
        }
        
        # Add temporal features if we have history
        if len(self.motion_magnitude_history) >= 5:
            stats['magnitude_std'] = np.std(list(self.motion_magnitude_history)[-5:])
            stats['density_std'] = np.std(list(self.motion_density_history)[-5:])
            stats['magnitude_trend'] = self._calculate_trend(list(self.motion_magnitude_history)[-10:])
        
        return {
            'flow_magnitude': magnitude,
            'flow_direction': direction,
            'motion_mask': motion_mask,
            'flow_x': flow_x,
            'flow_y': flow_y,
            'stats': stats,
        }
    
    def _empty_features(self, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Return empty feature set for first frame."""
        h, w = shape
        return {
            'flow_magnitude': np.zeros((h, w), dtype=np.float32),
            'flow_direction': np.zeros((h, w), dtype=np.float32),
            'motion_mask': np.zeros((h, w), dtype=np.uint8),
            'flow_x': np.zeros((h, w), dtype=np.float32),
            'flow_y': np.zeros((h, w), dtype=np.float32),
            'stats': {
                'avg_magnitude': 0.0,
                'max_magnitude': 0.0,
                'motion_density': 0.0,
                'motion_area': 0,
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate linear trend (positive = increasing, negative = decreasing).
        Simple linear regression slope.
        """
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Linear regression: y = mx + b
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def get_temporal_features(self) -> Optional[np.ndarray]:
        """
        Get temporal feature sequence for model input.
        
        Returns:
            Array of shape (history_length, n_features) suitable for LSTM input.
            Returns None if insufficient history.
        """
        if len(self.motion_magnitude_history) < self.history_length:
            return None
        
        # Stack temporal features
        features = []
        for i in range(self.history_length):
            frame_features = [
                self.motion_magnitude_history[i],
                self.motion_density_history[i],
                self.motion_direction_history[i],
            ]
            features.append(frame_features)
        
        return np.array(features, dtype=np.float32)
    
    def get_spatial_features(self) -> Optional[np.ndarray]:
        """
        Get spatial motion heatmap for CNN input.
        
        Returns:
            Array of shape (history_length, H, W) containing motion magnitude maps.
            Returns None if insufficient history.
        """
        # This would be implemented if using CNN-based model
        # For now, focusing on temporal features for LSTM
        pass
    
    def reset(self):
        """Clear all history."""
        self.frame_history.clear()
        self.gray_history.clear()
        self.motion_magnitude_history.clear()
        self.motion_density_history.clear()
        self.motion_direction_history.clear()
        logger.info("Motion feature extractor reset")
