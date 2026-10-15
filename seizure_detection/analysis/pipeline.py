"""
Multi-stage seizure detection pipeline - Refactored v2.0

Implements a 4-stage cascading detection strategy:
1. Motion Detection (Stage 1) - Fast motion screening (gate)
2. Generic Activity Detection (Stage 2) - Motion-based activity analysis
3. Motion LSTM Model (Stage 3) - Motion-based seizure detection (optional)
4. Pose-based Model (Stage 4) - Pose-based seizure detection (optional)

Pure detection logic - returns structured results.
Alert/notification logic handled by caller (server).
"""
import numpy as np
from datetime import datetime, time
from typing import List, Dict, Optional
from pathlib import Path

from .motion_detector import MotionDetector
from .motion_activity_detector import MotionActivityDetector
from .onnx_detector import ONNXSeizureDetector
from ..config import SeizureDetectionConfig
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DetectionResult:
    """
    Structured detection result from pipeline.
    
    Contains results from all stages plus seizure state tracking.
    """
    def __init__(self):
        self.timestamp: str = ""
        self.time_weight: float = 1.0
        self.stage_reached: int = 0
        
        # Stage 1: Motion Detection
        self.motion_detected: bool = False
        self.motion_degree: float = 0.0
        self.motion_threshold: float = 0.0
        
        # Stage 2: Generic Activity
        self.activity_detected: bool = False
        self.activity_confidence: float = 0.0
        self.activity_threshold: float = 0.0
        self.activity_details: Optional[Dict] = None
        
        # Stage 3: Motion LSTM Model
        self.motion_model_detected: bool = False
        self.motion_model_confidence: float = 0.0
        self.motion_model_threshold: float = 0.0
        self.motion_model_ready: bool = False
        
        # Motion persistence detection (applies to motion model output)
        self.motion_persistence_detected: bool = False
        self.motion_persistence_score: float = 0.0
        self.motion_persistence_duration: float = 0.0
        
        # Stage 4: Pose-based Model
        self.pose_model_detected: bool = False
        self.pose_model_confidence: float = 0.0
        self.pose_model_threshold: float = 0.0
        self.pose_model_ready: bool = False
        
        # Persistence Detection (sustained high confidence)
        self.persistence_detected: bool = False
        self.persistence_score: float = 0.0  # Fraction of window above threshold
        self.persistence_duration: float = 0.0  # How long high confidence has persisted
        
        # Seizure state tracking
        self.seizure_onset: bool = False
        self.seizure_offset: bool = False
        self.seizure_start_time: Optional[str] = None
        self.seizure_current_duration: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        # Helper to convert numpy types to Python native types
        def convert_value(val):
            if val is None:
                return val
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            if isinstance(val, (np.bool_, bool)):
                return bool(val)
            if isinstance(val, (np.integer, int)):
                return int(val)
            if isinstance(val, (np.floating, float)):
                return float(val)
            return val
        
        return {
            'timestamp': self.timestamp,
            'time_weight': convert_value(self.time_weight),
            'stage_reached': convert_value(self.stage_reached),
            'stage_1_motion': {
                'detected': convert_value(self.motion_detected),
                'degree': convert_value(self.motion_degree),
                'threshold': convert_value(self.motion_threshold),
            },
            'stage_2_activity': {
                'detected': convert_value(self.activity_detected),
                'confidence': convert_value(self.activity_confidence),
                'threshold': convert_value(self.activity_threshold),
                'details': self.activity_details,
            },
            'stage_3_motion_model': {
                'detected': convert_value(self.motion_model_detected),
                'confidence': convert_value(self.motion_model_confidence),
                'threshold': convert_value(self.motion_model_threshold),
                'ready': convert_value(self.motion_model_ready),
                'persistence_detected': convert_value(self.motion_persistence_detected),
                'persistence_score': convert_value(self.motion_persistence_score),
                'persistence_duration': convert_value(self.motion_persistence_duration),
            },
            'stage_4_pose_model': {
                'detected': convert_value(self.pose_model_detected),
                'confidence': convert_value(self.pose_model_confidence),
                'threshold': convert_value(self.pose_model_threshold),
                'ready': convert_value(self.pose_model_ready),
            },
            'seizure_state': {
                'onset': convert_value(self.seizure_onset),
                'offset': convert_value(self.seizure_offset),
                'start_time': self.seizure_start_time,
                'current_duration': convert_value(self.seizure_current_duration),
            }
        }


class MultiStageDetector:
    """
    Multi-stage seizure detection pipeline.
    
    Pure detection logic - returns DetectionResult with all stage outputs.
    Caller (server) decides when to alert based on results.
    """
    
    def __init__(
        self,
        config: SeizureDetectionConfig = SeizureDetectionConfig(),
        motion_model_path: Optional[Path] = None,
        motion_model_threshold: float = 0.8,
        pose_model_path: Optional[Path] = None,
        pose_model_threshold: float = 0.5,
        enable_time_weighting: bool = True,
        seizure_offset_threshold: int = 30,  # DEPRECATED: kept for compatibility
        intra_seizure_max_gap: float = 30.0,  # Max gap between spasms during seizure (seconds)
        seizure_end_threshold: float = 60.0  # Min time without activity to declare seizure ended (seconds)
    ):
        """
        Initialize multi-stage detector.
        
        Args:
            config: Detection configuration
            motion_model_path: Path to motion LSTM model (Stage 3, optional)
            motion_model_threshold: Confidence threshold for motion model
            pose_model_path: Path to pose-based model (Stage 4, optional)
            pose_model_threshold: Confidence threshold for pose model
            enable_time_weighting: Enable time-of-day risk adjustment
            seizure_offset_threshold: DEPRECATED - kept for compatibility
            intra_seizure_max_gap: Maximum gap between spasms during active seizure (seconds)
            seizure_end_threshold: Minimum time without activity to declare seizure ended (seconds)
        """
        self.config = config
        self.enable_time_weighting = enable_time_weighting
        self.seizure_offset_threshold = seizure_offset_threshold  # Kept for compatibility
        self.intra_seizure_max_gap = intra_seizure_max_gap
        self.seizure_end_threshold = seizure_end_threshold
        
        # Stage 1: Motion Detection (always enabled)
        self.motion_detector = MotionDetector()
        
        # Stage 2: Motion-based Activity Detection (always enabled, NO POSES NEEDED)
        self.activity_detector = MotionActivityDetector(config)
        
        # Stage 3: Motion LSTM Model (optional)
        self.motion_model = None
        self.motion_model_threshold = motion_model_threshold
        if motion_model_path and motion_model_path.exists():
            try:
                from .motion_model_detector import MotionModelDetector
                self.motion_model = MotionModelDetector(
                    model_path=motion_model_path,
                    confidence_threshold=motion_model_threshold
                )
                logger.info(f"✓ Stage 3: Motion LSTM enabled (threshold: {motion_model_threshold})")
            except Exception as e:
                logger.error(f"Failed to load motion model: {e}")
        
        # Stage 4: Pose-based Model (optional)
        self.pose_model = None
        self.pose_model_threshold = pose_model_threshold
        if pose_model_path and pose_model_path.exists():
            try:
                self.pose_model = ONNXSeizureDetector(
                    model_path=pose_model_path,
                    sequence_length=config.HYPERMOTOR_SEQUENCE_LENGTH,
                    confidence_threshold=pose_model_threshold
                )
                logger.info(f"✓ Stage 4: Pose model enabled (threshold: {pose_model_threshold})")
            except Exception as e:
                logger.error(f"Failed to load pose model: {e}")
        
        # Warn if no detection models enabled
        if not self.motion_model and not self.pose_model:
            logger.warning("No detection models enabled (Stage 3/4) - using generic activity detection only")
        
        # Persistence detector for sustained high-confidence detection (POSE MODEL)
        # Based on empirical analysis: seizures show 99.1% persistence above 0.9 vs 4.9% for normal
        from .persistence_detector import PersistenceDetector
        self.persistence_detector = PersistenceDetector(
            window_seconds=30.0,  # Analyze last 30 seconds
            confidence_threshold=0.7,  # Count predictions above 0.7 as "high"
            persistence_threshold=0.80,  # 80% of window must be high confidence
            min_duration=15.0  # Must persist for 15 seconds before alerting
        )
        logger.info("✓ Persistence detector enabled (30s window, 80% threshold, 15s min duration)")
        
        # Separate persistence detector for MOTION MODEL (Stage 3)
        # Uses same empirically-derived parameters to filter motion model false positives
        self.motion_persistence_detector = PersistenceDetector(
            window_seconds=30.0,
            confidence_threshold=0.7,
            persistence_threshold=0.80,
            min_duration=15.0
        )
        logger.info("✓ Motion persistence detector enabled (filters motion model false positives)")
        
        # Seizure state tracking
        self.in_seizure_state: bool = False
        self.seizure_start_time: Optional[datetime] = None
        self.frames_since_detection: int = 0
        
        # Rhythmic pattern tracking for hypermotor seizures
        # Hypermotor seizures show: spasm -> calm (4-20s) -> spasm -> longer calm -> ...
        self.detection_timestamps: List[datetime] = []  # Track when spasms occur
        self.last_detection_time: Optional[datetime] = None
        self.spasm_intervals: List[float] = []  # Intervals between spasms in seconds
        self.max_interval_during_seizure: float = 0.0  # Track longest calm period
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'motion_triggers': 0,
            'activity_triggers': 0,
            'motion_model_triggers': 0,
            'pose_model_triggers': 0,
        }
    
    def _get_time_weight(self) -> float:
        """
        Get risk weight based on time of day.
        
        Early morning (5-8 AM) is high-risk period for hypermotor seizures.
        
        Returns:
            Weight multiplier (1.0-1.5)
        """
        if not self.enable_time_weighting:
            return 1.0
        
        now = datetime.now().time()
        
        # High risk: 5:00 AM - 8:00 AM
        high_risk_start = time(5, 0)
        high_risk_end = time(8, 0)
        
        if high_risk_start <= now < high_risk_end:
            return 1.5  # 50% higher sensitivity
        
        # Moderate risk: 4:00 AM - 5:00 AM, 8:00 AM - 9:00 AM
        elif time(4, 0) <= now < high_risk_start:
            return 1.25
        elif high_risk_end <= now < time(9, 0):
            return 1.25
        
        # Normal risk: rest of day
        return 1.0
    
    def set_pose_threshold(self, threshold: float):
        """
        Dynamically update pose model threshold (e.g., for post-wake sensitivity boost).
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        if threshold < 0.0 or threshold > 1.0:
            logger.warning(f"Invalid pose threshold {threshold}, must be 0.0-1.0")
            return
        
        old_threshold = self.pose_model_threshold
        self.pose_model_threshold = threshold
        
        # Also update the ONNX detector's threshold if it exists
        if self.pose_model:
            self.pose_model.confidence_threshold = threshold
        
        if old_threshold != threshold:
            logger.info(f"Pose model threshold updated: {old_threshold:.2f} → {threshold:.2f}")
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: datetime,
        show_overlay: bool = False
    ) -> tuple[DetectionResult, np.ndarray]:
        """
        Analyze frame through multi-stage pipeline.
        
        Extracts poses internally, runs all detection stages,
        and returns structured result.
        
        Args:
            frame: Video frame (BGR format)
            timestamp: Frame timestamp
            show_overlay: Whether to draw pose skeleton overlay for visualization
        
        Returns:
            Tuple of (DetectionResult object, drawn frame with overlays)
        """
        result = DetectionResult()
        # Handle both datetime and float timestamps
        if isinstance(timestamp, datetime):
            result.timestamp = timestamp.isoformat()
            timestamp_float = timestamp.timestamp()
        else:
            # timestamp is already a float (Unix timestamp)
            result.timestamp = datetime.fromtimestamp(timestamp).isoformat()
            timestamp_float = timestamp
        result.time_weight = self._get_time_weight()
        
        self.stats['frames_processed'] += 1
        
        # Extract pose landmarks ONLY if needed by Stage 4 (pose model)
        # Stage 2 now uses motion-based detection (NO POSES NEEDED)
        # Only extract poses if pose model is enabled to save processing time
        landmarks = None
        drawn_frame = frame  # Keep track of frame with drawings
        if self.pose_model:
            try:
                from .pose_detection import process_frame as process_frame_pose
                drawn_frame, pose_telemetry = process_frame_pose(frame, do_draw=show_overlay)
                pose_data = pose_telemetry.get("pose", {})
                if pose_data:
                    landmarks = pose_data.get("landmarks", [])
            except Exception as e:
                logger.error(f"Pose extraction error in pipeline: {e}")
        
        # Stage 1: Motion Detection (GATE - bypassed during active seizure)
        motion_detected, motion_degree = self.motion_detector.detect_motion(frame)
        result.motion_detected = motion_detected
        result.motion_degree = motion_degree
        result.motion_threshold = self.config.MOTION_DEGREE_THRESHOLD
        result.stage_reached = 1
        
        # Bypass gate if already in seizure (to track through calm periods)
        if not motion_detected and not self.in_seizure_state:
            self._update_seizure_state(result, any_detection=False, timestamp=timestamp)
            return result, drawn_frame  # Stop here - no motion and not in seizure
        
        self.stats['motion_triggers'] += 1
        
        # Stage 2: Motion-based Activity Detection (GATE - bypassed during active seizure)
        activity_result = self.activity_detector.process_frame(frame, timestamp)
        
        activity_confidence = activity_result['confidence']
        weighted_activity_conf = activity_confidence * result.time_weight
        
        result.activity_confidence = activity_confidence
        result.activity_threshold = self.config.CONFIDENCE_THRESHOLD
        result.activity_detected = weighted_activity_conf >= self.config.CONFIDENCE_THRESHOLD
        result.activity_details = activity_result
        result.stage_reached = 2
        
        # Bypass gate if already in seizure (to track through calm periods)
        if not result.activity_detected and not self.in_seizure_state:
            self._update_seizure_state(result, any_detection=False, timestamp=timestamp)
            return result, drawn_frame  # Stop here - activity not suspicious and not in seizure
        
        self.stats['activity_triggers'] += 1
        
        # Stage 3: Motion LSTM Model (if enabled)
        if self.motion_model:
            motion_model_result = self.motion_model.process_frame(frame)
            
            result.motion_model_ready = motion_model_result.get('ready', False)
            result.motion_model_confidence = motion_model_result.get('confidence', 0.0)
            result.motion_model_threshold = self.motion_model_threshold
            
            if result.motion_model_ready:
                weighted_motion_conf = result.motion_model_confidence * result.time_weight
                result.motion_model_detected = weighted_motion_conf >= self.motion_model_threshold
                result.stage_reached = 3
                
                # Apply motion persistence detector (filters false positives)
                motion_persistence_result = self.motion_persistence_detector.add_prediction(
                    result.motion_model_confidence,
                    timestamp_float
                )
                
                # Store motion persistence metrics
                result.motion_persistence_score = motion_persistence_result['persistence_score']
                result.motion_persistence_duration = motion_persistence_result['duration']
                result.motion_persistence_detected = motion_persistence_result['seizure_detected']
                
                if result.motion_persistence_detected:
                    self.stats['motion_model_triggers'] += 1
        
        # Stage 4: Pose-based Model (if enabled)
        if self.pose_model and self.pose_model.is_ready():
            self.pose_model.add_pose(landmarks, timestamp)
            pose_result = self.pose_model.predict()
            
            if pose_result and pose_result.get('ready'):
                result.pose_model_ready = True
                result.pose_model_confidence = pose_result['probability']
                result.pose_model_threshold = self.pose_model_threshold
                
                # Add prediction to persistence detector (needs float timestamp)
                persistence_result = self.persistence_detector.add_prediction(
                    result.pose_model_confidence, 
                    timestamp_float
                )
                
                # Store persistence metrics
                result.persistence_score = persistence_result['persistence_score']
                result.persistence_duration = persistence_result['duration']
                result.persistence_detected = persistence_result['seizure_detected']
                
                if persistence_result['seizure_detected']:
                    logger.info(
                        f"⚡ Sustained seizure detected: {persistence_result['duration']:.1f}s duration, "
                        f"{persistence_result['persistence_score']:.1%} persistence "
                        f"({persistence_result['high_confidence_count']}/{persistence_result['window_size']} predictions)"
                    )
                
                weighted_pose_conf = result.pose_model_confidence * result.time_weight
                result.pose_model_detected = weighted_pose_conf >= self.pose_model_threshold
                result.stage_reached = 4
                
                if result.pose_model_detected:
                    self.stats['pose_model_triggers'] += 1
        
        # Determine if seizure detected - require BOTH models to agree
        # This prevents false positives from either model alone
        any_detection = (
            result.persistence_detected and 
            result.motion_persistence_detected
        )
        
        # Update seizure state tracking
        self._update_seizure_state(result, any_detection=any_detection, timestamp=timestamp)
        
        return result, drawn_frame
    
    def _update_seizure_state(
        self,
        result: DetectionResult,
        any_detection: bool,
        timestamp: datetime
    ):
        """Update seizure state tracking and populate result fields."""
        
        if any_detection:
            self.frames_since_detection = 0
            
            # Track spasm timing for rhythmic pattern analysis
            if self.last_detection_time is not None:
                # Calculate interval - handle both timestamp types
                if isinstance(timestamp, datetime) and isinstance(self.last_detection_time, datetime):
                    interval = (timestamp - self.last_detection_time).total_seconds()
                elif isinstance(timestamp, datetime):
                    interval = timestamp.timestamp() - self.last_detection_time
                elif isinstance(self.last_detection_time, datetime):
                    interval = timestamp - self.last_detection_time.timestamp()
                else:
                    interval = timestamp - self.last_detection_time
                
                self.spasm_intervals.append(interval)
                # Keep last 20 intervals
                if len(self.spasm_intervals) > 20:
                    self.spasm_intervals.pop(0)
            
            self.last_detection_time = timestamp
            self.detection_timestamps.append(timestamp)
            # Keep last 50 detection times
            if len(self.detection_timestamps) > 50:
                self.detection_timestamps.pop(0)
            
            if not self.in_seizure_state:
                # NEW SEIZURE ONSET
                result.seizure_onset = True
                result.seizure_offset = False
                self.in_seizure_state = True
                self.seizure_start_time = timestamp if isinstance(timestamp, datetime) else datetime.fromtimestamp(timestamp)
                result.seizure_start_time = self.seizure_start_time.isoformat()
                result.seizure_current_duration = 0.0
                
                # Reset rhythmic pattern tracking
                self.max_interval_during_seizure = 0.0
                
                logger.warning(f"🚨 SEIZURE ONSET detected at {timestamp}")
            else:
                # SEIZURE CONTINUING (spasm detected)
                result.seizure_onset = False
                result.seizure_offset = False
                result.seizure_start_time = self.seizure_start_time.isoformat()
                # Calculate duration - handle both timestamp types
                if isinstance(timestamp, datetime):
                    result.seizure_current_duration = (timestamp - self.seizure_start_time).total_seconds()
                else:
                    result.seizure_current_duration = timestamp - self.seizure_start_time.timestamp()
                
                # Log if this spasm came after a notable calm period
                if self.spasm_intervals:
                    last_interval = self.spasm_intervals[-1]
                    if last_interval > 4.0:  # Calm period of 4+ seconds
                        logger.info(
                            f"Rhythmic pattern: spasm after {last_interval:.1f}s calm "
                            f"(duration: {result.seizure_current_duration:.1f}s)"
                        )
                
                logger.debug(f"Seizure ongoing (duration: {result.seizure_current_duration:.1f}s)")
        
        else:
            # No detection this frame
            self.frames_since_detection += 1
            
            if self.in_seizure_state:
                # In seizure but currently in calm period
                result.seizure_start_time = self.seizure_start_time.isoformat()
                # Calculate duration - handle both timestamp types
                if isinstance(timestamp, datetime):
                    result.seizure_current_duration = (timestamp - self.seizure_start_time).total_seconds()
                else:
                    result.seizure_current_duration = timestamp - self.seizure_start_time.timestamp()
                
                # Calculate time since last spasm
                time_since_last_spasm = 0.0
                if self.last_detection_time:
                    # Handle both timestamp types
                    if isinstance(timestamp, datetime) and isinstance(self.last_detection_time, datetime):
                        time_since_last_spasm = (timestamp - self.last_detection_time).total_seconds()
                    elif isinstance(timestamp, datetime):
                        time_since_last_spasm = timestamp.timestamp() - self.last_detection_time
                    elif isinstance(self.last_detection_time, datetime):
                        time_since_last_spasm = timestamp - self.last_detection_time.timestamp()
                    else:
                        time_since_last_spasm = timestamp - self.last_detection_time
                    
                    # Track the longest calm period during this seizure
                    if time_since_last_spasm > self.max_interval_during_seizure:
                        self.max_interval_during_seizure = time_since_last_spasm
                
                # TWO-LEVEL TIMEOUT LOGIC:
                # 1. Intra-seizure calm periods: normal rhythmic pattern between spasms
                # 2. Seizure ended: extended period with no activity
                
                # Calculate adaptive threshold based on seizure progression
                # Early seizure: shorter intervals between spasms
                # Late seizure: longer intervals (gradually increasing)
                intra_seizure_threshold = self.intra_seizure_max_gap
                
                # If we've seen rhythmic patterns, adapt threshold dynamically
                if len(self.spasm_intervals) >= 3:
                    # Use 2x the average interval, capped at configured max
                    avg_interval = sum(self.spasm_intervals[-5:]) / len(self.spasm_intervals[-5:])
                    intra_seizure_threshold = min(self.intra_seizure_max_gap * 1.5, avg_interval * 2)
                
                # Seizure end threshold from config
                seizure_end_threshold = self.seizure_end_threshold
                
                if time_since_last_spasm >= seizure_end_threshold:
                    # SEIZURE OFFSET - Long period of calm, seizure truly ended
                    result.seizure_onset = False
                    result.seizure_offset = True
                    self.in_seizure_state = False
                    
                    final_duration = result.seizure_current_duration
                    num_spasms = len(self.detection_timestamps)
                    avg_interval = sum(self.spasm_intervals) / len(self.spasm_intervals) if self.spasm_intervals else 0
                    
                    logger.warning(
                        f"✅ SEIZURE OFFSET detected at {timestamp} "
                        f"(duration: {final_duration:.1f}s, spasms: {num_spasms}, "
                        f"avg interval: {avg_interval:.1f}s, max gap: {self.max_interval_during_seizure:.1f}s)"
                    )
                    
                    # Clear state after offset
                    self.seizure_start_time = None
                    result.seizure_start_time = None
                    result.seizure_current_duration = None
                    self.detection_timestamps.clear()
                    self.spasm_intervals.clear()
                    self.last_detection_time = None
                    self.max_interval_during_seizure = 0.0
                    
                elif time_since_last_spasm >= intra_seizure_threshold:
                    # Longer calm period - log warning but don't end seizure yet
                    result.seizure_onset = False
                    result.seizure_offset = False
                    logger.info(
                        f"⚠️  Extended calm period: {time_since_last_spasm:.1f}s "
                        f"(threshold: {intra_seizure_threshold:.1f}s, "
                        f"seizure duration: {result.seizure_current_duration:.1f}s)"
                    )
                else:
                    # Normal intra-seizure calm period (expected rhythmic pattern)
                    result.seizure_onset = False
                    result.seizure_offset = False
                    if time_since_last_spasm >= 4.0:  # Only log if calm period is notable
                        logger.debug(
                            f"Calm period during seizure: {time_since_last_spasm:.1f}s "
                            f"(duration: {result.seizure_current_duration:.1f}s)"
                        )
            else:
                # Not in seizure state, no detection
                result.seizure_onset = False
                result.seizure_offset = False
                result.seizure_start_time = None
                result.seizure_current_duration = None
    
    def should_alert(self, result: DetectionResult, strategy: str = "onset_only") -> bool:
        """
        Helper method to determine if result should trigger alert.
        
        Can be called by server/orchestration layer to decide alerting.
        Provides different strategies for combining model results.
        
        Args:
            result: DetectionResult from analyze_frame()
            strategy: How to handle alerts
                - "onset_only": Alert only on NEW seizure onset
                - "onset_and_ongoing": Alert on onset + periodic updates
                - "onset_and_offset": Alert on onset AND when seizure ends
                - "any_model": Alert if ANY model detects (Stage 3 or 4)
                - "both_models": Alert only if BOTH models detect
                - "motion_only": Alert only on motion model (Stage 3)
                - "pose_only": Alert only on pose model (Stage 4)
        
        Returns:
            True if should alert based on strategy
        """
        # Must pass Stage 1 (motion threshold)
        # Stage 2 (activity) only required if threshold > 0
        if not result.motion_detected:
            return False
        if result.activity_threshold > 0 and not result.activity_detected:
            return False
        
        if strategy == "onset_only":
            return result.seizure_onset
        
        elif strategy == "onset_and_ongoing":
            # Alert on onset, then every 60 seconds during seizure
            if result.seizure_onset:
                return True
            if result.seizure_current_duration and result.seizure_current_duration > 0:
                return result.seizure_current_duration % 60 < 1  # Every 60 seconds
            return False
        
        elif strategy == "onset_and_offset":
            return result.seizure_onset or result.seizure_offset
        
        elif strategy == "any_model":
            return result.motion_model_detected or result.pose_model_detected
        
        elif strategy == "both_models":
            # Only if both are ready and both detect
            both_ready = result.motion_model_ready and result.pose_model_ready
            both_detected = result.motion_model_detected and result.pose_model_detected
            return both_ready and both_detected
        
        elif strategy == "motion_only":
            return result.motion_model_ready and result.motion_model_detected
        
        elif strategy == "pose_only":
            return result.pose_model_ready and result.pose_model_detected
        
        else:
            logger.warning(f"Unknown alert strategy: {strategy}, using 'onset_only'")
            return result.seizure_onset
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        total_frames = self.stats['frames_processed']
        
        return {
            **self.stats,
            'efficiency': {
                'motion_pass_rate': self.stats['motion_triggers'] / total_frames if total_frames > 0 else 0,
                'activity_pass_rate': self.stats['activity_triggers'] / self.stats['motion_triggers'] if self.stats['motion_triggers'] > 0 else 0,
            },
            'models_enabled': {
                'motion_lstm': self.motion_model is not None,
                'pose_model': self.pose_model is not None,
            },
            'time_weighting_enabled': self.enable_time_weighting,
            'seizure_state': {
                'in_seizure': self.in_seizure_state,
                'current_duration': (datetime.now() - self.seizure_start_time).total_seconds() if self.seizure_start_time else None
            }
        }
    
    def reset(self):
        """Reset all detectors and state."""
        self.motion_detector.reset()
        self.activity_detector.reset()
        if self.motion_model:
            self.motion_model.reset() if hasattr(self.motion_model, 'reset') else None
        if self.pose_model:
            self.pose_model.reset()
        
        self.in_seizure_state = False
        self.seizure_start_time = None
        self.frames_since_detection = 0
        
        logger.info("Multi-stage detector reset")


# Keep old DetectionStage class for backward compatibility
class DetectionStage:
    """Detection pipeline stages."""
    MOTION = 1
    ACTIVITY = 2
    MOTION_MODEL = 3
    POSE_MODEL = 4
    # Aliases for backward compatibility
    CUSTOM_MODEL = 4
