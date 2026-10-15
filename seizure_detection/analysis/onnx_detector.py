"""
ONNX-based custom seizure detector.

Loads the trained hypermotor seizure detection model
and performs inference on pose sequences.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import deque
from datetime import datetime

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ONNXSeizureDetector:
    """
    Custom seizure detector using ONNX model.
    
    Analyzes sequences of pose keypoints to detect hypermotor seizures.
    """
    
    def __init__(
        self,
        model_path: Path,
        sequence_length: int = 30,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize ONNX detector.
        
        Args:
            model_path: Path to ONNX model file
            sequence_length: Number of frames in sequence (30 = 6 seconds at 5 FPS)
            confidence_threshold: Threshold for seizure detection
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Pose sequence buffer
        self.pose_buffer = deque(maxlen=sequence_length)
        self.timestamps = deque(maxlen=sequence_length)
        
        # Model session
        self.session = None
        self._load_model()
        
        # Detection state
        self.last_prediction = 0.0
        self.consecutive_detections = 0
        
    def _load_model(self):
        """Load ONNX model for inference."""
        try:
            import onnxruntime
            
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return
            
            # Create inference session
            self.session = onnxruntime.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            
            logger.info(f"✓ Loaded ONNX model: {self.model_path}")
            logger.info(f"  Input: {self.session.get_inputs()[0].name}")
            logger.info(f"  Output: {self.session.get_outputs()[0].name}")
            
        except ImportError:
            logger.error("onnxruntime not installed. Install: pip install onnxruntime")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self.session is not None
    
    def add_pose(
        self,
        landmarks: Optional[List[Dict]],
        timestamp: datetime
    ):
        """
        Add pose landmarks to sequence buffer.
        
        Args:
            landmarks: List of detected pose landmarks (from MediaPipe)
            timestamp: Frame timestamp
        """
        if not landmarks or len(landmarks) == 0:
            # Add zeros for missing pose (33 MediaPipe landmarks)
            keypoints = np.zeros((33, 3), dtype=np.float32)
        else:
            # Extract all 33 MediaPipe landmarks
            keypoints = self._extract_keypoints(landmarks[0])
        
        self.pose_buffer.append(keypoints)
        self.timestamps.append(timestamp)
    
    def _extract_keypoints(self, person) -> np.ndarray:
        """
        Extract all 33 MediaPipe pose landmarks.
        
        MediaPipe Pose landmarks (33 total):
        0-10: Face (nose, eyes, ears, mouth)
        11-16: Arms (shoulders, elbows, wrists)
        17-22: Hands (finger tips, pinkies, indexes, thumbs)
        23-28: Legs (hips, knees, ankles)
        29-32: Feet (heels, foot indexes)
        
        Args:
            person: List of landmark dictionaries from MediaPipe
        
        Returns:
            Array of shape (33, 3) with x, y, z coordinates
        """
        keypoints = np.zeros((33, 3), dtype=np.float32)
        
        # MediaPipe returns a list of 33 landmarks
        # Each landmark has x, y, z, and visibility
        if isinstance(person, list):
            for idx, landmark in enumerate(person):
                if idx < 33:  # Ensure we don't exceed 33 landmarks
                    keypoints[idx] = [
                        landmark.get('x', 0.0),
                        landmark.get('y', 0.0),
                        landmark.get('z', 0.0)
                    ]
        
        return keypoints
    
    def predict(self) -> Optional[Dict]:
        """
        Run inference on current pose sequence.
        
        Returns:
            Dictionary with prediction results, or None if not ready
        """
        if not self.is_ready():
            return None
        
        if len(self.pose_buffer) < self.sequence_length:
            return {
                'ready': False,
                'buffer_size': len(self.pose_buffer),
                'required_size': self.sequence_length
            }
        
        try:
            # Prepare input: (1, 30, 33, 3)
            sequence = np.array(list(self.pose_buffer), dtype=np.float32)
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            outputs = self.session.run(
                [output_name],
                {input_name: sequence}
            )
            
            # Get prediction probability
            probability = float(outputs[0][0][0])
            self.last_prediction = probability
            
            # Track consecutive detections
            if probability >= self.confidence_threshold:
                self.consecutive_detections += 1
            else:
                self.consecutive_detections = 0
            
            # Detection requires confidence above threshold
            detected = probability >= self.confidence_threshold
            
            return {
                'ready': True,
                'detected': detected,
                'probability': probability,
                'threshold': self.confidence_threshold,
                'consecutive_detections': self.consecutive_detections,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None
    
    def reset(self):
        """Reset detector state."""
        self.pose_buffer.clear()
        self.timestamps.clear()
        self.last_prediction = 0.0
        self.consecutive_detections = 0
        logger.info("ONNX detector reset")
