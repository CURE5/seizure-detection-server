"""
Configuration management for seizure detection system.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_LOGS_DIR = BASE_DIR / "data_logs"
SEIZURE_LOGS_DIR = BASE_DIR / "seizure_logs"
MEDICATION_LOGS_DIR = BASE_DIR / "medication_logs"
HR_LOGS_DIR = BASE_DIR / "hr_logs"
VIDEO_DIR = BASE_DIR / os.getenv('RECORDING_DIRECTORY', 'recorded_video')
MEDIAPIPE_MODELS_DIR = BASE_DIR / "mediapipe_models"

# Ensure directories exist
for directory in [DATA_LOGS_DIR, SEIZURE_LOGS_DIR, MEDICATION_LOGS_DIR, HR_LOGS_DIR, VIDEO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Video settings
VIDEO_PATH = os.getenv('VIDEO_PATH', '')
CODEC = os.getenv('CODEC', 'H264')
VIDEO_CONTAINER = os.getenv('VIDEO_CONTAINER', 'mp4')
USE_FFMPEG = os.getenv('USE_FFMPEG', 'False').lower() == 'true'
REENCODE_WITH_FFMPEG = os.getenv('REENCODE_WITH_FFMPEG', 'False').lower() == 'true'

# Camera settings
DEFAULT_RESOLUTION = (640, 360)
DAY_CAMERA_NUM = 0      # Normal camera (color, daylight)
NIGHT_CAMERA_NUM = 1    # IR camera (grayscale, works in darkness)

# Motion detection settings - read from environment for easy tuning
MOTION_THRESHOLD = int(os.getenv('MOTION_THRESHOLD', '25'))
MOTION_DEGREE_THRESHOLD = float(os.getenv('MOTION_DEGREE_THRESHOLD', '0.025'))
SLEEP_TIMEOUT_SECONDS = 4 * 60  # 4 minutes

# FPS settings
class FPSConfig:
    # Pose-inference rate while the subject is active. This is a clinical
    # sensitivity knob: higher = more frames analysed for seizure motion, more
    # CPU/heat. Trimmed 5.0 -> 4.0 (-20% inference) to ease thermal throttling.
    # Revert to 5.0 if detection sensitivity regresses.
    ACTIVE = 4.0
    SLEEP_TRANSITION = 1.0  # Recent motion detected
    SLEEP_LIGHT = 0.5       # Some time since motion
    SLEEP_DEEP = 0.2        # Long time since motion
    
    TRANSITION_THRESHOLD = 120  # 2 minutes
    LIGHT_THRESHOLD = 300       # 5 minutes

# MediaPipe settings
class MediaPipeConfig:
    POSE_MODEL = str(MEDIAPIPE_MODELS_DIR / "pose_landmarker.task")
    FACE_MODEL = str(MEDIAPIPE_MODELS_DIR / "face_landmarker.task")
    
    NUM_POSES = 4
    MIN_POSE_DETECTION_CONFIDENCE = 0.8
    MIN_POSE_PRESENCE_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    OUTPUT_SEGMENTATION_MASKS = True
    
    MIN_FACE_PRESENCE_CONFIDENCE = 0.05
    MIN_FACE_DETECTION_CONFIDENCE = 0.05
    NUM_FACES = 1
    OUTPUT_FACE_BLENDSHAPES = False

# Pose detection backend selection
FORCE_MEDIAPIPE = os.getenv('FORCE_MEDIAPIPE', 'False').lower() == 'true'

# Detection-alert behavior. When true, AI-detected events (visual detector or
# HR detector) push a notification ONLY — no seizure log entry, no auto-record,
# no is_seizing state flip. Caregiver confirms manually via POST /sz when the
# event is real. Independent from SHADOW_MODE (which silences notifications).
DETECTION_ALERT_NOTIFY_ONLY = os.getenv('DETECTION_ALERT_NOTIFY_ONLY', 'true').lower() == 'true'

# Hailo AI settings
class HailoConfig:
    MODEL_PATH = os.getenv('HAILO_MODEL_PATH', '/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef')
    DETECTION_THRESHOLD = float(os.getenv('HAILO_THRESHOLD', '0.5'))
    NMS_IOU_THRESHOLD = float(os.getenv('HAILO_NMS_THRESHOLD', '0.7'))
    
    # Custom trained model for patient-specific seizure detection
    CUSTOM_MODEL_PATH = os.getenv('HAILO_CUSTOM_MODEL_PATH', '')
    USE_CUSTOM_MODEL = os.getenv('USE_CUSTOM_MODEL', 'False').lower() == 'true'

# Seizure detection settings
class SeizureDetectionConfig:
    # Generic detector settings
    HISTORY_LENGTH = 30  # frames (6 seconds at 5 FPS)
    MOVEMENT_THRESHOLD = 0.1
    RHYTHMIC_FREQUENCY_RANGE = (2, 5)  # Hz
    VELOCITY_THRESHOLD = 0.15
    ACCELERATION_THRESHOLD = 0.25
    CONFIDENCE_THRESHOLD = float(os.getenv('ACTIVITY_THRESHOLD', '0.7'))  # Stage 2 activity detection
    MOTION_DEGREE_THRESHOLD = MOTION_DEGREE_THRESHOLD  # From module level (Stage 1)
    
    # Custom model settings
    USE_CUSTOM_MODEL = os.getenv('USE_CUSTOM_MODEL', 'False').lower() == 'true'
    CUSTOM_MODEL_PATH = os.getenv('CUSTOM_MODEL_PATH', 
                                   'models/hypermotor_seizure_detector.onnx')
    CUSTOM_MODEL_TYPE = os.getenv('CUSTOM_MODEL_TYPE', 'onnx')
    
    # Ensemble mode (combine generic + custom)
    ENSEMBLE_MODE = os.getenv('ENSEMBLE_MODE', 'False').lower() == 'true'
    CUSTOM_MODEL_WEIGHT = 0.8  # Higher weight for custom model
    GENERIC_DETECTOR_WEIGHT = 0.2
    
    # Hypermotor spasm specific parameters
    SEIZURE_TYPE = 'hypermotor'
    HYPERMOTOR_SEQUENCE_LENGTH = 30  # 6 seconds at 5 FPS
    HYPERMOTOR_KEY_JOINTS = [9, 10, 15, 16]  # Wrists and ankles (most active)

# API settings
class APIConfig:
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')
    FRAME_QUEUE_SIZE = 10
    # Affects only the live preview stream sent to the dashboard, not the
    # analysis frame or the recorded video. 70 is visually close to 85 at a
    # noticeably lower encode cost.
    JPEG_QUALITY = 70

# Logging settings
class LoggingConfig:
    LOG_FILE = str(BASE_DIR / "logs" / "seizure_detection.log")
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT = 5

# Garmin / heart-rate detector settings
class GarminHRConfig:
    ENABLED = os.getenv('ENABLE_GARMIN_HR', 'false').lower() == 'true'
    ADDRESS: Optional[str] = os.getenv('GARMIN_HR_ADDRESS') or None
    NAME_FILTER = os.getenv('GARMIN_HR_NAME_FILTER', 'garmin')

    SCAN_TIMEOUT_SEC = float(os.getenv('GARMIN_HR_SCAN_TIMEOUT_SEC', '10'))
    RESCAN_INTERVAL_SEC = float(os.getenv('GARMIN_HR_RESCAN_INTERVAL_SEC', '30'))
    RECONNECT_DELAY_SEC = float(os.getenv('GARMIN_HR_RECONNECT_DELAY_SEC', '5'))
    CONNECT_TIMEOUT_SEC = float(os.getenv('GARMIN_HR_CONNECT_TIMEOUT_SEC', '20'))
    # On Linux, run `bluetoothctl disconnect <addr>` before each connect attempt
    # to clear stale BlueZ state from a prior process. Best-effort, errors ignored.
    CLEAR_BLUEZ_ON_RETRY = os.getenv('GARMIN_HR_CLEAR_BLUEZ_ON_RETRY', 'true').lower() == 'true'
    # If connected but no HR notification arrives for this many seconds, force
    # a reconnect. Catches BlueZ phantom-connected hangs.
    NOTIFICATION_STALE_SEC = float(os.getenv('GARMIN_HR_NOTIFICATION_STALE_SEC', '30'))
    # Hard cap on BleakClient.disconnect() so a stuck teardown can't freeze
    # the reconnect loop.
    DISCONNECT_TIMEOUT_SEC = float(os.getenv('GARMIN_HR_DISCONNECT_TIMEOUT_SEC', '5'))

    BASELINE_WINDOW_SEC = int(os.getenv('HR_BASELINE_WINDOW_SEC', '600'))
    DELTA_THRESHOLD = int(os.getenv('HR_DELTA_THRESHOLD', '15'))
    ABSOLUTE_THRESHOLD = int(os.getenv('HR_ABSOLUTE_THRESHOLD', '100'))
    SUSTAIN_SEC = int(os.getenv('HR_SUSTAIN_SEC', '20'))
    STALE_SEC = int(os.getenv('HR_STALE_SEC', '10'))
    ALERT_COOLDOWN_SEC = int(os.getenv('HR_ALERT_COOLDOWN_SEC', '60'))


# Internet connectivity
class ConnectivityConfig:
    CHECK_INTERVAL = 2 * 60  # 2 minutes
    RETRIES = 3
    RETRY_DELAY = 10  # seconds
    PING_HOST = "8.8.8.8"
    PING_TIMEOUT = 3

# Ensure log directory exists
Path(LoggingConfig.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
