"""
Tests for seizure detector
"""
from datetime import datetime
from seizure_detection.analysis import SeizureDetector


class TestSeizureDetector:
    """Test seizure detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SeizureDetector()
    
    def test_initialization(self):
        """Test detector initializes correctly."""
        assert self.detector.detection_score == 0.0
        assert len(self.detector.pose_history) == 0
        assert self.detector.last_detection_time is None
    
    def test_insufficient_data(self):
        """Test analysis with insufficient data."""
        result = self.detector.analyze()
        
        assert result['detected'] is False
        assert result['confidence'] == 0.0
    
    def test_add_pose_data_empty(self):
        """Test adding empty pose data."""
        self.detector.add_pose_data(None, datetime.now())
        assert len(self.detector.pose_history) == 0
        
        self.detector.add_pose_data([], datetime.now())
        assert len(self.detector.pose_history) == 0
    
    def test_add_pose_data_valid(self):
        """Test adding valid pose data."""
        landmarks = [{
            'LEFT_WRIST': {'x': 0.5, 'y': 0.5, 'z': 0.0},
            'RIGHT_WRIST': {'x': 0.5, 'y': 0.5, 'z': 0.0}
        }]
        
        self.detector.add_pose_data(landmarks, datetime.now())
        assert len(self.detector.pose_history) == 1
    
    def test_reset(self):
        """Test detector reset."""
        landmarks = [{
            'LEFT_WRIST': {'x': 0.5, 'y': 0.5, 'z': 0.0}
        }]
        
        self.detector.add_pose_data(landmarks, datetime.now())
        self.detector.detection_score = 0.8
        
        self.detector.reset()
        
        assert len(self.detector.pose_history) == 0
        assert self.detector.detection_score == 0.0
