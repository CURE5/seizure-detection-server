"""
Tests for motion detector
"""
import numpy as np
from seizure_detection.analysis import MotionDetector
from seizure_detection.utils.states import SleepMode


class TestMotionDetector:
    """Test motion detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MotionDetector()
    
    def test_initialization(self):
        """Test detector initializes correctly."""
        assert self.detector.motion_degree == 0.0
        assert self.detector.sleep_mode == SleepMode.ACTIVE
        assert self.detector.last_frame is None
    
    def test_first_frame(self):
        """Test processing first frame."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        motion_detected, degree = self.detector.detect_motion(frame)
        
        assert motion_detected is False
        assert degree == 0.0
        assert self.detector.last_frame is not None
    
    def test_no_motion(self):
        """Test identical frames (no motion)."""
        frame1 = np.zeros((360, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((360, 640, 3), dtype=np.uint8)
        
        self.detector.detect_motion(frame1)
        motion_detected, degree = self.detector.detect_motion(frame2)
        
        assert not motion_detected
        assert degree < 0.01
    
    def test_motion_detected(self):
        """Test motion detection with different frames."""
        frame1 = np.zeros((360, 640, 3), dtype=np.uint8)
        frame2 = np.ones((360, 640, 3), dtype=np.uint8) * 255
        
        self.detector.detect_motion(frame1)
        motion_detected, degree = self.detector.detect_motion(frame2)
        
        assert motion_detected
        assert degree > 0.5
    
    def test_adaptive_fps(self):
        """Test adaptive FPS calculation."""
        # Active mode
        from seizure_detection.config import FPSConfig
        fps = self.detector.get_adaptive_fps()
        assert fps == FPSConfig.ACTIVE
        
        # Note: Testing sleep mode transitions requires time manipulation
        # which is beyond this basic test
    
    def test_reset(self):
        """Test detector reset."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.detector.detect_motion(frame)
        
        self.detector.reset()
        
        assert self.detector.last_frame is None
        assert self.detector.motion_degree == 0.0
        assert self.detector.sleep_mode == SleepMode.ACTIVE
