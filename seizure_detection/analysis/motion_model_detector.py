"""
Motion-based LSTM seizure detector using trained model.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from ..utils.logger import setup_logger
from .motion_features import MotionFeatureExtractor

logger = setup_logger(__name__)


class MotionModelDetector:
    """
    LSTM-based seizure detector using motion features.
    
    Uses a trained ONNX model to classify motion patterns as seizure/normal.
    """
    
    def __init__(
        self,
        model_path: str = "models/motion_seizure_detector.onnx",
        sequence_length: int = 30,
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize motion model detector.
        
        Args:
            model_path: Path to ONNX model file
            sequence_length: Number of frames in sequence (default: 30 = 6s at 5 FPS)
            confidence_threshold: Threshold for seizure detection (0-1)
        """
        self.model_path = Path(model_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize ONNX runtime session
        self.session: Optional[ort.InferenceSession] = None
        self._load_model()
        
        # Motion feature extractor
        self.feature_extractor = MotionFeatureExtractor(history_length=sequence_length)
        
        # Statistics
        self.total_predictions = 0
        self.seizure_detections = 0
        self.last_prediction_time: Optional[datetime] = None
        self.last_confidence: float = 0.0
        
    def _load_model(self) -> bool:
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            logger.error("onnxruntime not available. Install with: pip install onnxruntime")
            return False
        
        if not self.model_path.exists():
            logger.error(f"Model not found: {self.model_path}")
            return False
        
        try:
            # Create ONNX runtime session
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            logger.info(f"✓ Loaded motion model: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and return seizure detection result.
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Dictionary with detection results:
            {
                'seizure_detected': bool,
                'confidence': float (0-1),
                'ready': bool (enough frames collected),
                'motion_features': dict
            }
        """
        if self.session is None:
            return {
                'seizure_detected': False,
                'confidence': 0.0,
                'ready': False,
                'error': 'Model not loaded'
            }
        
        # Extract motion features from frame
        motion_features = self.feature_extractor.process_frame(frame)
        
        # Check if we have enough frames for prediction
        if len(self.feature_extractor.motion_magnitude_history) < self.sequence_length:
            return {
                'seizure_detected': False,
                'confidence': 0.0,
                'ready': False,
                'frames_collected': len(self.feature_extractor.motion_magnitude_history),
                'frames_needed': self.sequence_length,
                'motion_features': motion_features
            }
        
        # Get temporal features for model input
        temporal_features = self.feature_extractor.get_temporal_features()
        
        if temporal_features is None:
            return {
                'seizure_detected': False,
                'confidence': 0.0,
                'ready': False,
                'error': 'Failed to extract temporal features'
            }
        
        try:
            # Prepare input: (1, 30, 3) = batch_size × sequence_length × features
            input_data = temporal_features.reshape(1, self.sequence_length, 3).astype(np.float32)
            
            # Run inference
            outputs = self.session.run(None, {'input': input_data})
            confidence = float(outputs[0][0][0])  # Sigmoid output
            
            # Update statistics
            self.total_predictions += 1
            self.last_prediction_time = datetime.now()
            self.last_confidence = confidence
            
            seizure_detected = confidence > self.confidence_threshold
            
            if seizure_detected:
                self.seizure_detections += 1
                logger.warning(f"🚨 Motion model detected seizure! Confidence: {confidence:.2%}")
            
            return {
                'seizure_detected': seizure_detected,
                'confidence': confidence,
                'ready': True,
                'motion_features': motion_features,
                'threshold': self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'seizure_detected': False,
                'confidence': 0.0,
                'ready': True,
                'error': str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'total_predictions': self.total_predictions,
            'seizure_detections': self.seizure_detections,
            'last_confidence': self.last_confidence,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'detection_rate': (
                self.seizure_detections / self.total_predictions 
                if self.total_predictions > 0 else 0.0
            ),
            'model_loaded': self.session is not None,
            'model_path': str(self.model_path),
            'sequence_length': self.sequence_length,
            'confidence_threshold': self.confidence_threshold
        }
    
    def reset(self):
        """Reset detector state."""
        self.feature_extractor = MotionFeatureExtractor(history_length=self.sequence_length)
        logger.info("Motion model detector reset")
