"""
Refactored FastAPI server integrating new modular architecture.
Version 2.0
"""
import asyncio
import base64
import cv2
import numpy as np
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Import new modular components
from seizure_detection.camera import CameraManager
from seizure_detection.analysis import MultiStageDetector, MotionDetector
from seizure_detection.recording import VideoRecorder
from seizure_detection.logging import (
    DataLogger,
    SeizureLogger,
    MedicationLogger,
    HRLogger,
)
from seizure_detection.notifications import NotificationManager
from seizure_detection.utils.logger import setup_logger
from seizure_detection.utils.connectivity import ConnectivityChecker
from seizure_detection.utils.health import HealthMonitor
from seizure_detection.utils.states import CameraMode, SleepMode
from seizure_detection.config import (
    APIConfig,
    VIDEO_DIR,
    GarminHRConfig,
    DETECTION_ALERT_NOTIFY_ONLY,
)

# Garmin HR monitor is imported lazily inside startup_event because it depends
# on `bleak`, which is only required when ENABLE_GARMIN_HR=true.

# Import pose detection processing modules
from seizure_detection.analysis.pose_detection import (
    process_frame as process_frame_pose,
)
from seizure_detection.analysis import MotionFeatureExtractor, MotionVisualizer
from seizure_detection.analysis.pose_detection import (
    setup as setup_pose_detection,
)
from seizure_detection.analysis.pose_detection import (
    release as release_pose_detection,
)

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI
app = FastAPI(title="Seizure Detection Server", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
camera_manager = CameraManager()
motion_detector = MotionDetector()
video_recorder = VideoRecorder()
data_logger = DataLogger()
seizure_logger = SeizureLogger()
medication_logger = MedicationLogger()
hr_logger = HRLogger()
connectivity_checker = ConnectivityChecker()
health_monitor = HealthMonitor()
notification_manager = NotificationManager()

# Garmin HR monitor (initialized at startup if ENABLE_GARMIN_HR=true).
# Typed as Any to avoid importing bleak at module load.
garmin_monitor: Any = None
hr_detector: Any = None
hr_detector_task: Optional[asyncio.Task] = None

# Initialize multi-stage detector with configurable models
use_motion_model = os.getenv("USE_MOTION_MODEL", "false").lower() == "true"
use_pose_model = os.getenv("USE_POSE_MODEL", "true").lower() == "true"

motion_model_path = None
if use_motion_model:
    motion_model_path = Path(os.getenv("MOTION_MODEL_PATH", "models/motion_seizure_detector.onnx"))

pose_model_path = None
if use_pose_model:
    pose_model_path = Path(os.getenv("POSE_MODEL_PATH", "models/hypermotor_detector.onnx"))

seizure_detector = MultiStageDetector(
    motion_model_path=motion_model_path,
    motion_model_threshold=float(os.getenv("MOTION_MODEL_THRESHOLD", "0.8")),
    pose_model_path=pose_model_path,
    pose_model_threshold=float(os.getenv("POSE_MODEL_THRESHOLD", "0.5")),
    enable_time_weighting=os.getenv("ENABLE_TIME_WEIGHTING", "true").lower() == "true",
    intra_seizure_max_gap=float(os.getenv("INTRA_SEIZURE_MAX_GAP", "30.0")),
    seizure_end_threshold=float(os.getenv("SEIZURE_END_THRESHOLD", "60.0"))
)

# Global state
frame_queue = asyncio.Queue(maxsize=APIConfig.FRAME_QUEUE_SIZE)
last_frame_sent: Optional[Dict] = None
# Timestamp of the most recent GET /frame poll. The dashboard polls /frame only
# while its live view is open (timer lives in the browser, cleared on unmount),
# so this tells us whether anyone is watching. The producer loop uses it to skip
# the throwaway JPEG/base64 encode when nobody is — the common case overnight.
last_frame_request_time: Optional[datetime] = None
# Keep producing preview frames for this long after the last /frame poll. Must
# exceed the dashboard's slowest poll interval (20s in sleep mode) so an open but
# idle dashboard still counts as watching.
STREAM_IDLE_SECONDS = 30
use_pose_detection = True
show_overlay = False
analyzing_video: Optional[str] = None
video_start_timestamp: Optional[float] = None  # Base timestamp for video analysis
video_frame_count: int = 0  # Frame counter for video analysis
motion_extractor: Optional[MotionFeatureExtractor] = None
motion_visualizer: Optional[MotionVisualizer] = None

is_seizing = False
seizure_started_at: Optional[datetime] = None
seizure_ended_at: Optional[datetime] = None
current_seizure_id: Optional[str] = None

# Shadow mode (testing without alerts)
shadow_mode = os.getenv("SHADOW_MODE", "false").lower() == "true"
auto_record = os.getenv("AUTO_RECORD_ON_DETECTION", "true").lower() == "true"

# Wake-up notification settings
notify_on_wakeup = os.getenv("NOTIFY_ON_WAKEUP", "false").lower() == "true"

# HR-spike notification toggle. Mirrors notify_on_wakeup — runtime-toggleable
# from the FE via /command. When false, the HR detector still runs (baseline,
# logging, /status) but suppresses notifications.
notify_on_hr_spike = os.getenv("NOTIFY_ON_HR_SPIKE", "false").lower() == "true"
wakeup_sleep_threshold_minutes = int(
    os.getenv("WAKEUP_SLEEP_THRESHOLD_MINUTES", "30")
)  # Minutes of no motion before considering "asleep"
last_wakeup_notification_time: Optional[datetime] = None
last_wakeup_time: Optional[datetime] = None  # Track most recent wake-up for seizure sensitivity

# Motion magnitude thresholds (used by motion feature extractor)
# Based on testing: Normal videos mean=0.16, Seizure videos mean=0.41
MOTION_SLEEP_THRESHOLD = 0.15   # Below this = sleeping (50th percentile normal)
WAKEUP_ACTIVITY_THRESHOLD = float(os.getenv("WAKEUP_ACTIVITY_THRESHOLD", "0.50"))  # Configurable wake threshold

# In-bed detection settings (optional person verification via skeleton tracking)
wakeup_require_in_bed = os.getenv("WAKEUP_REQUIRE_IN_BED", "false").lower() == "true"
person_detection_window_minutes = int(os.getenv("PERSON_DETECTION_WINDOW_MINUTES", "60"))
last_person_detected_time: Optional[datetime] = None  # Last time skeleton tracking detected person

# Post-wake seizure sensitivity boost (hypermotor seizures often start after waking)
POST_WAKE_SENSITIVITY_MINUTES = int(os.getenv("POST_WAKE_SENSITIVITY_MINUTES", "15"))
POST_WAKE_POSE_THRESHOLD = float(os.getenv("POST_WAKE_POSE_THRESHOLD", "0.3"))  # Lower = more sensitive
NORMAL_POSE_THRESHOLD = float(os.getenv("POSE_MODEL_THRESHOLD", "0.5"))  # Normal threshold

# Alert cooldown to prevent spam
last_alert_time: Optional[datetime] = None
alert_cooldown_seconds = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))


def has_valid_person_detection(pose_telemetry: Optional[Dict]) -> bool:
    """
    Check if pose telemetry contains valid person detection.

    Args:
        pose_telemetry: Pose detection result from process_frame_pose()

    Returns:
        True if a person is detected with sufficient confidence
    """
    if not pose_telemetry:
        return False

    try:
        pose_data = pose_telemetry.get("pose", {})
        if not pose_data:
            return False

        # Check if landmarks exist (either full landmarks or specific landmarks)
        landmarks = pose_data.get("landmarks", [])
        specific_landmarks = pose_data.get("specific")

        # If we have landmarks array with at least one person
        if landmarks and len(landmarks) > 0:
            # Check that we have a reasonable number of visible landmarks
            # MediaPipe returns 33 landmarks, we need at least 8-10 visible ones
            first_person_landmarks = landmarks[0]
            if isinstance(first_person_landmarks, list) and len(first_person_landmarks) >= 10:
                # Count how many landmarks have decent visibility
                visible_count = sum(
                    1 for lm in first_person_landmarks
                    if isinstance(lm, dict) and lm.get("visibility", 0) > 0.5
                )
                return visible_count >= 5  # At least 5 visible landmarks

        # Alternative: check specific landmarks (eyes and wrists)
        if specific_landmarks and len(specific_landmarks) > 0:
            return True  # If specific landmarks extracted, person was detected

        return False

    except Exception as e:
        logger.debug(f"Error checking person detection: {e}")
        return False


async def handle_hr_alert(alert: Dict) -> None:
    """Heart-rate alert path. Independent from visual detector.

    Not gated by SHADOW_MODE — that flag silences the unreliable visual
    detector, but HR is treated as a trustworthy independent signal with its
    own kill switch (`NOTIFY_ON_HR_SPIKE`). Still suppressed while visual is
    actively in a seizure state, to avoid double-alerting on the same event.
    Cooldown lives in `HRDetector`.
    """
    if is_seizing:
        logger.info(
            "HR alert suppressed (visual detector already in seizure state): "
            "bpm=%s baseline=%s trigger=%s",
            alert.get("bpm"),
            alert.get("baseline_bpm"),
            alert.get("trigger"),
        )
        return

    logger.warning("=" * 70)
    logger.warning(
        "🚨 HR ALERT TRIGGERED 🚨 (notify_only=%s)",
        DETECTION_ALERT_NOTIFY_ONLY,
    )
    logger.warning(
        "bpm=%s baseline=%s delta=%s sustained=%.1fs trigger=%s",
        alert.get("bpm"),
        alert.get("baseline_bpm"),
        alert.get("delta"),
        alert.get("sustained_sec", 0.0),
        alert.get("trigger"),
    )
    logger.warning("=" * 70)

    if not notify_on_hr_spike:
        logger.info(
            "HR notification suppressed (notify_on_hr_spike=False); "
            "detector still recording event."
        )
    else:
        try:
            results = await notification_manager.send_alert(alert)
            success = sum(1 for v in results.values() if v)
            logger.info(f"✓ Sent HR notifications: {success}/{len(results)} backends")
        except Exception as e:
            logger.error(f"Failed to send HR notifications: {e}")

    # Auto-actions hook — gated by DETECTION_ALERT_NOTIFY_ONLY. When false,
    # this is where auto-record into recorded_video/pending/, set is_seizing,
    # etc. would wire in. Intentionally not implemented yet.
    if not DETECTION_ALERT_NOTIFY_ONLY:
        logger.info(
            "DETECTION_ALERT_NOTIFY_ONLY=false but HR auto-actions not "
            "implemented; alert sent as notification only."
        )


async def handle_seizure_alert(result: Dict):
    """
    Handle seizure detection alert.

    Actions:
    1. Auto-start video recording
    2. Create seizure log entry
    3. Send notifications (unless shadow mode)
    4. Update global state
    """
    global is_seizing, seizure_started_at, current_seizure_id, video_recorder

    logger.warning("=" * 70)
    logger.warning(
        "🚨 SEIZURE ALERT TRIGGERED 🚨 (notify_only=%s)",
        DETECTION_ALERT_NOTIFY_ONLY,
    )
    logger.warning("=" * 70)

    # Auto-start recording — skipped in notify-only mode (caregiver confirms
    # the event manually; no auto-action on unreliable AI detection).
    recording_started = False
    if (
        auto_record
        and not video_recorder.is_recording()
        and not DETECTION_ALERT_NOTIFY_ONLY
    ):
        try:
            video_recorder.start_recording()
            recording_started = True
            logger.info("✓ Auto-started video recording")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")

    # Extract detection details from result
    stage_reached = result.get('stage_reached', 0)
    seizure_state = result.get('seizure_state', {})
    
    # Get confidence from highest stage that detected
    confidence = 0.0
    if result.get('stage_4_pose_model', {}).get('detected'):
        confidence = result.get('stage_4_pose_model', {}).get('confidence', 0.0)
    elif result.get('stage_3_motion_model', {}).get('detected'):
        confidence = result.get('stage_3_motion_model', {}).get('confidence', 0.0)
    elif result.get('stage_2_activity', {}).get('detected'):
        confidence = result.get('stage_2_activity', {}).get('confidence', 0.0)
    
    # Create seizure log entry using POST API endpoint
    now = datetime.now(timezone.utc)
    seizure_data = {
        "date": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "type": "Hypermotor spasm",
        "intensity": "2",
        "notes": (
            f"AI Auto-detected. "
            f"Confidence: {confidence:.2%}, "
            f"Stage: {stage_reached}, "
            f"Alert type: {result.get('alert_type', 'unknown')}"
        ),
        "video": "",
        "duration": "",
        "durationSeconds": 0,
        "aiDetected": True,
    }

    # Create seizure log entry. Skipped in test mode, shadow mode, and
    # notify-only mode. Notify-only is the production path for an unreliable
    # AI detector — caregiver confirms via POST /sz if the event is real.
    if (
        not analyzing_video
        and not shadow_mode
        and not DETECTION_ALERT_NOTIFY_ONLY
    ):
        try:
            event = await seizure_logger.create(seizure_data)
            current_seizure_id = event.get("id")
            logger.info(
                f"✓ Created AI-detected seizure log entry: {current_seizure_id}"
            )
        except Exception as e:
            logger.error(f"Failed to create seizure log: {e}")
    elif DETECTION_ALERT_NOTIFY_ONLY:
        logger.info(
            "⚠️  NOTIFY-ONLY: Skipped seizure log creation "
            "(caregiver confirms manually via POST /sz)"
        )
    else:
        current_seizure_id = f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if analyzing_video:
            logger.info(
                f"⚠️  TEST MODE (analyzing {analyzing_video}): Skipped seizure log creation"
            )
        elif shadow_mode:
            logger.info(
                "⚠️  SHADOW MODE: Skipped seizure log creation (detection ran normally)"
            )

    # Update global state. In notify-only mode we never flip is_seizing — the
    # event isn't system-confirmed, just flagged for human review. The offset
    # cleanup at the detector loop guards on `is_seizing or current_seizure_id`
    # so a no-op flip here means no-op cleanup later.
    if not DETECTION_ALERT_NOTIFY_ONLY:
        is_seizing = True
        seizure_started_at = datetime.fromisoformat(result["timestamp"])

    # Send notifications (skip in shadow mode, always send when analyzing video for testing)
    if analyzing_video and not shadow_mode:
        # Test mode: send alerts for testing notifications
        alert_data = {
            "timestamp": result["timestamp"],
            "confidence": confidence,
            "custom_probability": confidence,
            "stage": stage_reached,
            "time_weight": result.get("time_weight", 1.0),
            "recording_started": "Yes" if recording_started else "No",
            "seizure_id": current_seizure_id,
            "alert_type": result.get("alert_type", "unknown"),
            "duration": seizure_state.get("current_duration")
        }

        try:
            notification_results = await notification_manager.send_alert(
                alert_data
            )
            success_count = sum(1 for v in notification_results.values() if v)
            logger.info(
                f"✓ Sent test notifications: {success_count}/"
                f"{len(notification_results)} backends"
            )
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    elif not shadow_mode and not analyzing_video:
        # Production mode: send real alerts
        alert_data = {
            "timestamp": result["timestamp"],
            "confidence": confidence,
            "custom_probability": confidence,
            "stage": stage_reached,
            "time_weight": result.get("time_weight", 1.0),
            "recording_started": "Yes" if recording_started else "No",
            "seizure_id": current_seizure_id,
            "alert_type": result.get("alert_type", "unknown"),
            "duration": seizure_state.get("current_duration")
        }

        try:
            notification_results = await notification_manager.send_alert(
                alert_data
            )
            success_count = sum(1 for v in notification_results.values() if v)
            logger.info(
                f"✓ Sent notifications: {success_count}/"
                f"{len(notification_results)} backends"
            )
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    else:
        logger.info("⚠️  SHADOW MODE: Notifications and logging suppressed (detection still running)")


def parse_datetime(date_str: str) -> datetime:
    """Parse ISO 8601 datetime string."""
    date_str = date_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _build_hr_overlay_state() -> Dict:
    """Package monitor + detector state for the HR overlay strip."""
    state: Dict[str, Any] = {
        "enabled": GarminHRConfig.ENABLED,
        "connected": False,
        "stale": True,
        "bpm": None,
        "contact": None,
        "baseline_bpm": None,
        "above_baseline": False,
        "above_absolute": False,
        "sustained_sec": 0.0,
        "recent": [],
        "delta_threshold": GarminHRConfig.DELTA_THRESHOLD,
        "absolute_threshold": GarminHRConfig.ABSOLUTE_THRESHOLD,
        "sustain_sec": GarminHRConfig.SUSTAIN_SEC,
    }
    if garmin_monitor is not None:
        state["connected"] = garmin_monitor.is_connected
        latest = garmin_monitor.latest
        if latest is not None:
            age = (datetime.now(timezone.utc) - latest.timestamp).total_seconds()
            state["bpm"] = latest.bpm
            state["contact"] = latest.sensor_contact
            state["stale"] = age > GarminHRConfig.STALE_SEC
    if hr_detector is not None:
        ds = hr_detector.state()
        state["baseline_bpm"] = ds.baseline_bpm
        state["above_baseline"] = ds.above_baseline
        state["above_absolute"] = ds.above_absolute
        state["sustained_sec"] = ds.sustained_sec
        state["recent"] = ds.recent
    return state


def get_status() -> Dict:
    """Get current system status - maintains original format for compatibility."""
    camera_state = camera_manager.get_state()
    
    # Get motion stats from motion extractor if available
    global motion_extractor
    motion_magnitude = 0.0
    if motion_extractor and len(motion_extractor.motion_magnitude_history) > 0:
        motion_magnitude = motion_extractor.motion_magnitude_history[-1]
    
    # Get time since last motion and sleep mode from motion detector (consistent source)
    time_since_last_motion = motion_detector.get_time_since_last_motion().total_seconds()
    
    # Determine sleep mode - use motion_detector's state for consistency
    # When analyzing video, override sleep mode to show as AWAKE
    if analyzing_video:
        sleep_mode = "ANALYZING"
    else:
        # Update sleep mode first to ensure fresh state
        detector_sleep_mode = motion_detector.update_sleep_mode()
        if detector_sleep_mode == SleepMode.ACTIVE:
            sleep_mode = "AWAKE"
        elif detector_sleep_mode == SleepMode.TRANSITIONING:
            sleep_mode = "AWAKE"  # Still considered awake during transition
        else:
            sleep_mode = "SLEEP"

    # Determine night mode from camera mode
    night_mode = camera_manager.mode == CameraMode.NIGHT

    # Heart-rate block. Detector fields (baseline_bpm, above_*, sustained_sec,
    # last_alert_at) are populated once HRDetector lands; placeholders for now.
    hr_block: Dict[str, Any] = {
        "enabled": GarminHRConfig.ENABLED,
        "connected": False,
        "bpm": None,
        "contact": None,
        "stale": True,
        "baseline_bpm": None,
        "above_baseline": False,
        "above_absolute": False,
        "sustained_sec": 0.0,
        "last_alert_at": None,
    }
    if garmin_monitor is not None:
        hr_block["connected"] = garmin_monitor.is_connected
        latest = garmin_monitor.latest
        if latest is not None:
            age = (datetime.now(timezone.utc) - latest.timestamp).total_seconds()
            hr_block["bpm"] = latest.bpm
            hr_block["contact"] = latest.sensor_contact
            hr_block["stale"] = age > GarminHRConfig.STALE_SEC
    if hr_detector is not None:
        hr_state = hr_detector.state()
        hr_block["baseline_bpm"] = (
            round(hr_state.baseline_bpm, 1) if hr_state.baseline_bpm is not None else None
        )
        hr_block["above_baseline"] = hr_state.above_baseline
        hr_block["above_absolute"] = hr_state.above_absolute
        hr_block["sustained_sec"] = round(hr_state.sustained_sec, 1)
        hr_block["last_alert_at"] = (
            hr_state.last_alert_at.replace(microsecond=0).isoformat()
            if hr_state.last_alert_at
            else None
        )

    return {
        "type": "status",
        "is_capturing": camera_manager.is_running(),
        "is_recording": video_recorder.is_recording(),
        "camera_src": camera_state['source'],
        "camera_options": camera_state["options"],
        "use_pose_detection": use_pose_detection,
        "analyzing_video": analyzing_video,
        "connected_clients": 0,  # WebSocket removed, always 0
        "sleep_mode": sleep_mode,
        "night_mode": night_mode,
        "motion_magnitude": float(f"{motion_magnitude:.3f}"),
        "time_since_last_motion": float(f"{time_since_last_motion:.1f}"),
        "show_overlay": show_overlay,
        "notify_on_wakeup": notify_on_wakeup,
        "notify_on_hr_spike": notify_on_hr_spike,
        "is_seizing": is_seizing,
        "current_seizure_id": current_seizure_id,
        "seizure_started_at": seizure_started_at.replace(microsecond=0).isoformat()
        if seizure_started_at
        else None,
        "seizure_ended_at": seizure_ended_at.replace(microsecond=0).isoformat()
        if seizure_ended_at
        else None,
        "hr": hr_block,
        "detection_notify_only": DETECTION_ALERT_NOTIFY_ONLY,
    }


@app.get("/")
async def get_home():
    """Home page with simple viewer."""
    return HTMLResponse(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Seizure Detection Server v2.0</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #camera-feed { max-width: 100%; border: 2px solid #333; }
            .status { margin: 20px 0; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>Seizure Detection Server v2.0</h1>
        <div>
            <img id="camera-feed" src="/frame" alt="Camera Feed" />
        </div>
        <div class="status">
            <h3>Status</h3>
            <pre id="status-display">Loading...</pre>
        </div>
        <script>
            const refreshInterval = 200;  // 5 FPS
            const img = document.getElementById('camera-feed');
            const statusDisplay = document.getElementById('status-display');

            async function fetchFrame() {
                try {
                    const response = await fetch('/frame');
                    if (response.ok) {
                        const data = await response.json();
                        if (data.frame) {
                            img.src = `data:image/jpeg;base64,${data.frame}`;
                        }
                    }
                } catch (error) {
                    console.error('Error fetching frame:', error);
                }
            }
            
            async function fetchStatus() {
                try {
                    const response = await fetch('/status');
                    if (response.ok) {
                        const data = await response.json();
                        statusDisplay.textContent = JSON.stringify(data, null, 2);
                    }
                } catch (error) {
                    console.error('Error fetching status:', error);
                }
            }

            setInterval(fetchFrame, refreshInterval);
            setInterval(fetchStatus, 1000);
        </script>
    </body>
    </html>
    """
    )


@app.get("/frame")
async def get_frame():
    """Get current video frame."""
    global last_frame_sent, last_frame_request_time
    # Mark that a client is watching, so the producer loop resumes encoding.
    last_frame_request_time = datetime.now()

    if frame_queue.empty():
        if last_frame_sent:
            return JSONResponse(content=last_frame_sent, status_code=200)
        return JSONResponse(
            content={"error": "No frame available"}, status_code=200
        )

    msg = await frame_queue.get()
    last_frame_sent = msg
    return JSONResponse(content=msg, status_code=200)


@app.get("/status")
async def get_status_endpoint():
    """Get current system status."""
    status = get_status()

    # Add pipeline statistics
    status["detection_stats"] = seizure_detector.get_statistics()
    status["shadow_mode"] = shadow_mode
    status[
        "notification_backends"
    ] = notification_manager.get_enabled_backends()
    
    # Add pipeline configuration
    status["pipeline"] = {
        "motion_model_enabled": use_motion_model,
        "pose_model_enabled": use_pose_model,
        "alert_strategy": os.getenv("ALERT_STRATEGY", "onset_only")
    }
    status["notify_on_wakeup"] = notify_on_wakeup
    status["notify_on_hr_spike"] = notify_on_hr_spike
    status["wakeup_sleep_threshold_minutes"] = wakeup_sleep_threshold_minutes
    status["wakeup_activity_threshold"] = WAKEUP_ACTIVITY_THRESHOLD
    status["wakeup_require_in_bed"] = wakeup_require_in_bed
    status["person_detection_window_minutes"] = person_detection_window_minutes
    status["last_person_detected"] = last_person_detected_time.isoformat() if last_person_detected_time else None
    
    # Add comprehensive detection diagnostics
    from seizure_detection.config import MOTION_DEGREE_THRESHOLD
    import time
    
    # Capture timestamp when diagnostics are read
    diagnostics_timestamp = time.time()
    
    # Get current detector state (fresh calculation each time)
    persistence_state = seizure_detector.persistence_detector.get_state() if hasattr(seizure_detector, 'persistence_detector') else {}
    motion_persistence_state = seizure_detector.motion_persistence_detector.get_state() if hasattr(seizure_detector, 'motion_persistence_detector') else {}
    
    # Get activity threshold from environment
    activity_threshold = float(os.getenv("ACTIVITY_THRESHOLD", "0.003"))
    
    status["diagnostics"] = {
        "captured_at": diagnostics_timestamp,
        "captured_at_iso": datetime.fromtimestamp(diagnostics_timestamp).isoformat(),
        "gates": {
            "motion": {
                "threshold": MOTION_DEGREE_THRESHOLD,
                "current_value": motion_detector.motion_degree,
                "enabled": True,
                "description": "Stage 1: Motion gate (frame difference)"
            },
            "activity": {
                "threshold": activity_threshold,
                "current_value": getattr(motion_extractor, 'activity_score', 0.0) if motion_extractor else 0.0,
                "enabled": activity_threshold > 0,
                "description": "Stage 2: Activity gate (motion magnitude)"
            }
        },
        "models": {
            "motion_lstm": {
                "enabled": use_motion_model,
                "confidence_threshold": 0.5,
                "description": "Stage 3: Motion LSTM model"
            },
            "pose_onnx": {
                "enabled": use_pose_model,
                "confidence_threshold": 0.5,
                "description": "Stage 4: Pose ONNX model"
            }
        },
        "persistence_filters": {
            "pose_persistence": {
                "window_seconds": 30,
                "confidence_threshold": 0.7,
                "persistence_threshold": 0.8,
                "min_duration_seconds": 15,
                "current_predictions": len(persistence_state.get('predictions', [])),
                "current_persistence_score": persistence_state.get('persistence_score', 0.0),
                "current_duration": persistence_state.get('duration', 0.0),
                "seizure_detected": persistence_state.get('seizure_detected', False),
                "description": "Filters pose model: requires 80% of 30s window > 0.7 confidence, min 15s duration"
            },
            "motion_persistence": {
                "window_seconds": 30,
                "confidence_threshold": 0.7,
                "persistence_threshold": 0.8,
                "min_duration_seconds": 15,
                "current_predictions": len(motion_persistence_state.get('predictions', [])),
                "current_persistence_score": motion_persistence_state.get('persistence_score', 0.0),
                "current_duration": motion_persistence_state.get('duration', 0.0),
                "seizure_detected": motion_persistence_state.get('seizure_detected', False),
                "description": "Filters motion LSTM: requires 80% of 30s window > 0.7 confidence, min 15s duration"
            }
        },
        "seizure_state": {
            "onset_threshold_seconds": 15,
            "offset_threshold_seconds": 60,
            "in_seizure": seizure_detector.in_seizure_state if hasattr(seizure_detector, 'in_seizure_state') else False,
            "current_duration": (datetime.now() - seizure_detector.seizure_start_time).total_seconds() if hasattr(seizure_detector, 'seizure_start_time') and seizure_detector.seizure_start_time else None,
            "description": "Seizure starts after 15s sustained detection, ends after 60s below threshold"
        },
        "alert_strategy": {
            "strategy": os.getenv("ALERT_STRATEGY", "onset_only"),
            "description": "onset_only: Alert only on new seizure start | all: Alert on onset, offset, updates"
        }
    }

    return JSONResponse(content=status, status_code=200)


@app.get("/notifications/stats")
async def get_notification_stats():
    """Get notification system statistics."""
    return JSONResponse(
        content=notification_manager.get_statistics(), status_code=200
    )


@app.get("/notifications/history")
async def get_notification_history(limit: int = 10):
    """Get recent notification history."""
    return JSONResponse(
        content={"history": notification_manager.get_alert_history(limit)},
        status_code=200,
    )


@app.get("/health")
async def get_health():
    """Get system health information."""
    health_info = health_monitor.get_system_info()
    is_healthy, message = health_monitor.check_health()

    return JSONResponse(
        content={**health_info, "is_healthy": is_healthy, "message": message},
        status_code=200 if is_healthy else 503,
    )


# Seizure endpoints
@app.get("/sz")
async def get_seizures(start: str, end: str):
    """Get seizure events in date range."""
    try:
        start_dt = parse_datetime(start)
        end_dt = parse_datetime(end)
        seizures = await seizure_logger.get_range(start_dt, end_dt)
        return {"events": seizures}
    except Exception as e:
        logger.error(f"Error getting seizures: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sz")
async def create_seizure_event(event_data: Dict[str, str]):
    """Create a new seizure event."""
    try:
        event = await seizure_logger.create(event_data)
        if event:
            return {
                "status": True,
                "message": "Seizure event created",
                "event": event,
            }
        raise HTTPException(
            status_code=400, detail="Failed to create seizure event"
        )
    except Exception as e:
        logger.error(f"Error creating seizure: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/sz/{event_id}")
async def update_seizure_event(
    event_id: str, updates: Dict[str, Any]
):
    """Update a seizure event."""
    try:
        updated = await seizure_logger.update(event_id, updates)
        if updated:
            return {
                "status": True,
                "message": "Seizure event updated",
                "event": updated,
            }
        raise HTTPException(status_code=404, detail="Seizure event not found")
    except Exception as e:
        logger.error(f"Error updating seizure: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/sz/{event_id}")
async def delete_seizure_event(event_id: str):
    """Delete a seizure event."""
    try:
        deleted = await seizure_logger.delete(event_id)
        if deleted:
            return {"message": "Seizure event deleted"}
        raise HTTPException(status_code=404, detail="Seizure event not found")
    except Exception as e:
        logger.error(f"Error deleting seizure: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Medication endpoints
@app.get("/medicine")
async def get_medications(start: str, end: str):
    """Get medication events in date range."""
    try:
        start_dt = parse_datetime(start)
        end_dt = parse_datetime(end)
        medications = await medication_logger.get_range(start_dt, end_dt)
        return {"events": medications}
    except Exception as e:
        logger.error(f"Error getting medications: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/medicine")
async def create_medication_event(event_data: Dict[str, str]):
    """Create a new medication event."""
    try:
        event = await medication_logger.create(event_data)
        if event:
            return {
                "status": True,
                "message": "Medication event created",
                "event": event,
            }
        raise HTTPException(
            status_code=400, detail="Failed to create medication event"
        )
    except Exception as e:
        logger.error(f"Error creating medication: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/medicine/{event_id}")
async def update_medication_event(
    event_id: str, updates: Dict[str, Optional[str]]
):
    """Update a medication event."""
    try:
        updated = await medication_logger.update(event_id, updates)
        if updated:
            return {
                "status": True,
                "message": "Medication event updated",
                "event": updated,
            }
        raise HTTPException(
            status_code=404, detail="Medication event not found"
        )
    except Exception as e:
        logger.error(f"Error updating medication: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/medicine/{event_id}")
async def delete_medication_event(event_id: str):
    """Delete a medication event."""
    try:
        deleted = await medication_logger.delete(event_id)
        if deleted:
            return {"message": "Medication event deleted"}
        raise HTTPException(
            status_code=404, detail="Medication event not found"
        )
    except Exception as e:
        logger.error(f"Error deleting medication: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Heart-rate samples endpoint. Columnar parallel arrays match the on-disk
# schema and keep the wire payload compact for plotting / training.
@app.get("/hr")
async def get_hr(start: str, end: str):
    """Get HR samples in [start, end). Returns columnar arrays.

    Response shape:
        {"t": [epoch_ms, ...], "bpm": [...], "rr": [...], "contact": [...]}

    Derived fields (baseline, above-threshold) are intentionally absent — they
    are recomputed at read time from samples + canonical event logs.
    """
    try:
        start_dt = parse_datetime(start)
        end_dt = parse_datetime(end)
        return await hr_logger.read_range(start_dt, end_dt)
    except Exception as e:
        logger.error(f"Error getting HR samples: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/hr/around/{event_id}")
async def get_hr_around(
    event_id: str,
    before_sec: float = 300.0,
    after_sec: float = 300.0,
):
    """Get HR samples around a seizure event, centered on its onset.

    `t_rel_sec` is seconds-from-onset (negative = before, positive = after),
    so windows from different events stack directly into a training tensor.
    """
    sz = await seizure_logger.get_by_id(event_id)
    if sz is None:
        raise HTTPException(status_code=404, detail="Seizure event not found")
    try:
        center = parse_datetime(sz["date"])
        samples = await hr_logger.window(center, before_sec, after_sec)
        return {"sz": sz, "samples": samples}
    except Exception as e:
        logger.error(f"Error getting HR around {event_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Data logs endpoint
@app.get("/logs")
async def get_logs(
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
):
    """Get data logs in date range."""
    try:
        logs = await data_logger.read_logs(start_date, end_date)
        return logs
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Command endpoint
@app.post("/cmd")
async def command_handler(command: Dict[str, Any]):
    """Handle control commands."""
    global use_pose_detection, show_overlay, analyzing_video, video_start_timestamp, video_frame_count
    global is_seizing, seizure_started_at, seizure_ended_at, current_seizure_id
    global notify_on_wakeup

    action = command.get("action")
    logger.info(f"Command received: {action}")

    try:
        if action == "startCapturing":
            if not camera_manager.is_running():
                await camera_manager.start()
            return {"status": "Capturing started"}

        elif action == "stopCapturing":
            if camera_manager.is_running():
                await camera_manager.stop()
                if video_recorder.is_recording():
                    await video_recorder.stop()
            
            # Reset all state when stopping camera
            analyzing_video = None  # Clear video analysis state (exits test mode)
            video_start_timestamp = None
            video_frame_count = 0
            
            # End any active seizure state
            if is_seizing or current_seizure_id:
                if current_seizure_id and not analyzing_video:
                    try:
                        await seizure_logger.end(current_seizure_id)
                        logger.info(f"✓ Closed seizure log entry on camera stop: {current_seizure_id}")
                    except Exception as e:
                        logger.error(f"Failed to close seizure log: {e}")
                
                is_seizing = False
                seizure_ended_at = datetime.now()
                current_seizure_id = None
                logger.info("✓ Seizure state cleared on camera stop")
            
            return {"status": "Capturing stopped"}

        elif action == "startRecording":
            if not video_recorder.is_recording():
                fps = motion_detector.get_adaptive_fps()
                frame_size = camera_manager.resolution
                await video_recorder.start(fps=fps, frame_size=frame_size)
            return {"status": "Recording started"}

        elif action == "stopRecording":
            if video_recorder.is_recording():
                output_file = await video_recorder.stop()
                return {
                    "status": "Recording stopped",
                    "file": str(output_file),
                }
            return {"status": "No recording in progress"}

        elif action == "night_mode":
            night_mode = command.get("value", False)
            await camera_manager.switch_mode(night_mode=night_mode)
            return {"status": f"Night mode: {night_mode}"}

        elif action == "show_overlay":
            show_overlay = command.get("value", False)
            return {"status": f"Show overlay: {show_overlay}"}

        elif action == "notify_on_wakeup":
            notify_on_wakeup = command.get("value", False)
            logger.info(f"Wake-up notifications: {notify_on_wakeup}")
            return {"status": f"Notify on wakeup: {notify_on_wakeup}"}
        elif action == "notify_on_hr_spike":
            global notify_on_hr_spike
            notify_on_hr_spike = bool(command.get("value", False))
            logger.info(f"HR-spike notifications: {notify_on_hr_spike}")
            return {"status": f"Notify on HR spike: {notify_on_hr_spike}"}

        elif action == "startPoseDetection":
            if not use_pose_detection:
                use_pose_detection = True
                logger.info("Pose detection enabled")
            return {"status": "Pose detection started"}

        elif action == "stopPoseDetection":
            if use_pose_detection:
                use_pose_detection = False
            return {"status": "Pose detection stopped"}

        elif action == "analyzeVideo":
            video_name = command.get("value")
            
            # If switching videos or starting fresh, clean up previous state
            if camera_manager.is_running():
                await camera_manager.stop()
                if video_recorder.is_recording():
                    await video_recorder.stop()
                
                # End any active seizure state from previous video/camera
                if is_seizing or current_seizure_id:
                    is_seizing = False
                    seizure_ended_at = datetime.now()
                    current_seizure_id = None
                    logger.info("✓ Seizure state cleared before analyzing new video")
            
            # Start new video analysis (enters test mode)
            analyzing_video = video_name
            video_start_timestamp = datetime.now().timestamp()  # Record start time for video playback
            video_frame_count = 0  # Reset frame counter
            show_overlay = True  # Automatically enable pose skeleton visualization for video analysis
            video_path = str(VIDEO_DIR / video_name)

            await camera_manager.start(video_path=video_path)
            return {"status": f"Analyzing video: {video_name}"}

        elif action == "seizure_started":
            is_seizing = True
            seizure_started_at = datetime.now()

            if not camera_manager.is_running():
                await camera_manager.start()
            if not video_recorder.is_recording():
                fps = motion_detector.get_adaptive_fps()
                await video_recorder.start(
                    fps=fps, frame_size=camera_manager.resolution
                )

            value = command.get("value", {})
            current_seizure_id = await seizure_logger.start(
                seizure_type=value.get("type", "Epileptic spasm"),
                intensity=value.get("intensity", "0"),
                notes=value.get("notes", ""),
                video="",
            )

            logger.warning(f"Seizure started: {current_seizure_id}")
            return {"status": "Seizure started", "id": current_seizure_id}

        elif action == "seizure_ended":
            is_seizing = False
            seizure_ended_at = datetime.now()

            if current_seizure_id:
                await seizure_logger.end(current_seizure_id)
                logger.info(f"Seizure ended: {current_seizure_id}")

            if video_recorder.is_recording():
                output_file = await video_recorder.stop()
                # TODO: Update seizure event with video filename

            seizure_id = current_seizure_id
            current_seizure_id = None

            return {"status": "Seizure ended", "id": seizure_id}

        elif action == "setControls":
            options = command.get("options", {})
            success = await camera_manager.set_controls(options)
            return {
                "status": "Controls set"
                if success
                else "Failed to set controls"
            }

        elif action == "testAlert":
            # Send a test alert to all configured notification backends
            logger.info("Sending test alert to notification backends")

            test_alert_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.95,
                "custom_probability": 0.95,
                "stage": "custom",
                "time_weight": 1.0,
                "recording_started": "No",
                "seizure_id": "test-alert",
                "is_test": True,
            }

            try:
                notification_results = await notification_manager.send_alert(
                    test_alert_data
                )
                success_count = sum(
                    1 for v in notification_results.values() if v
                )

                logger.info(
                    f"✓ Test alert sent: {success_count}/"
                    f"{len(notification_results)} backends succeeded"
                )

                return {
                    "status": "Test alert sent",
                    "backends": notification_results,
                    "success_count": success_count,
                    "total_backends": len(notification_results),
                }
            except Exception as e:
                logger.error(f"Failed to send test alert: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to send test alert: {str(e)}",
                )

        else:
            return {"status": "Unknown action", "action": action}

    except Exception as e:
        logger.exception(f"Error handling command {action}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_frames():
    """Main video processing loop."""
    global use_pose_detection, show_overlay
    global notify_on_wakeup
    global last_wakeup_notification_time
    global last_person_detected_time
    global last_alert_time

    logger.info("Starting video processing loop")
    
    # Frame counter for skipping expensive processing in sleep mode
    frame_counter = 0
    last_processing_time = datetime.now()

    try:
        while True:
            if not camera_manager.is_running():
                await asyncio.sleep(1)
                continue

            # Capture frame
            frame = camera_manager.capture_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            frame_counter += 1

            # Detect motion (always run at reasonable rate to catch wake-ups)
            motion_detected, motion_degree = motion_detector.detect_motion(
                frame
            )
            
            # Get adaptive FPS for expensive processing (pose detection, analysis)
            processing_fps = motion_detector.get_adaptive_fps()
            
            # Motion detection always runs at minimum 1 FPS to catch wake-ups quickly
            # But expensive processing runs at adaptive rate
            motion_check_fps = max(1.0, processing_fps)
            
            # Determine if we should do expensive processing this frame
            # Only do full processing at the adaptive rate
            # When analyzing video, always process (disable sleep mode throttling)
            time_since_processing = (datetime.now() - last_processing_time).total_seconds()
            should_process_frame = analyzing_video or time_since_processing >= (1.0 / processing_fps)

            # Process with signal pipeline if enabled
            analysis_result = {}
            processed_frame = frame.copy()
            pose_landmarks = None
            motion_features = None
            # Initialize detection result (will be populated if detection runs)
            from seizure_detection.analysis.pipeline import DetectionResult
            detection_result = DetectionResult()
            detection_result.stage_reached = 0

            # Extract motion features (always - needed for detection)
            global motion_extractor, motion_visualizer
            if motion_extractor is None:
                motion_extractor = MotionFeatureExtractor(history_length=30)
            if motion_visualizer is None:
                motion_visualizer = MotionVisualizer()
            
            # ALWAYS extract motion features to detect wake-ups from sleep
            try:
                motion_features = motion_extractor.process_frame(frame)
                # Convert numpy types to Python types for JSON serialization
                motion_stats = motion_features.get("stats", {})
                analysis_result["motion"] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in motion_stats.items()
                }

                # Person detection logic (for in-bed verification)
                # Only check when there's significant movement - either:
                # 1. Potential wake-up (movement spike after sleep)
                # 2. Initial activity (baby getting into bed)
                if wakeup_require_in_bed and len(motion_extractor.motion_magnitude_history) > 0:
                    current_motion = motion_extractor.motion_magnitude_history[-1]

                    # Check for significant movement that warrants person detection check
                    significant_movement = current_motion > WAKEUP_ACTIVITY_THRESHOLD

                    if significant_movement:
                        try:
                            # Check for person presence when we detect significant movement
                            _, pose_telemetry = process_frame_pose(frame, do_draw=False)
                            if has_valid_person_detection(pose_telemetry):
                                # Person detected - update timestamp
                                last_person_detected_time = datetime.now(timezone.utc)
                                logger.debug("Person detected during movement check")
                            else:
                                # No person detected during movement
                                # Could be: 1) Movement blur (skeleton fails during motion), or
                                #          2) Person left bed
                                # Don't immediately invalidate - log for debugging
                                if last_person_detected_time is not None:
                                    minutes_since = (
                                        datetime.now(timezone.utc) - last_person_detected_time
                                    ).total_seconds() / 60
                                    logger.debug(
                                        f"Movement detected but no person found "
                                        f"(last seen {minutes_since:.1f} min ago - likely movement blur)"
                                    )
                                else:
                                    logger.debug(
                                        "Movement detected but no person history "
                                        "(baby may not be in bed yet)"
                                    )
                        except Exception as e:
                            logger.debug(f"Pose detection during movement check error: {e}")

                # Wake detection using motion magnitude history
                if notify_on_wakeup and len(motion_extractor.motion_magnitude_history) > 0:
                        current_motion = motion_extractor.motion_magnitude_history[-1]

                        # Calculate average motion over last N minutes to detect sleep
                        sleep_minutes = wakeup_sleep_threshold_minutes
                        frames_to_check = int(sleep_minutes * 60 * processing_fps)  # minutes * seconds * fps

                        if len(motion_extractor.motion_magnitude_history) >= frames_to_check:
                            recent_motion = list(motion_extractor.motion_magnitude_history)[-frames_to_check:]
                            avg_motion = sum(recent_motion) / len(recent_motion)

                            # Build wake-up conditions
                            wake_conditions = [
                                avg_motion < MOTION_SLEEP_THRESHOLD,  # Was sleeping (low motion)
                                current_motion > WAKEUP_ACTIVITY_THRESHOLD,  # Now active (configurable threshold)
                            ]

                            # Optionally verify person was detected in bed (via skeleton tracking)
                            # Trust any detection within the configured window
                            if wakeup_require_in_bed:
                                person_was_in_bed = False

                                if last_person_detected_time is not None:
                                    minutes_since_detection = (
                                        datetime.now(timezone.utc) - last_person_detected_time
                                    ).total_seconds() / 60
                                    person_was_in_bed = minutes_since_detection <= person_detection_window_minutes

                                    if not person_was_in_bed:
                                        logger.info(
                                            f"Wake-up suppressed: No recent person detection "
                                            f"(last seen {minutes_since_detection:.1f} min ago, "
                                            f"threshold: {person_detection_window_minutes} min)"
                                        )
                                else:
                                    logger.info("Wake-up suppressed: No person detection history available")

                                wake_conditions.append(person_was_in_bed)

                            # Detect wake-up if all conditions met
                            if all(wake_conditions):
                                now = datetime.now(timezone.utc)
                                
                                # Prevent duplicate notifications within 5 minutes
                                if last_wakeup_notification_time is None or (
                                    now - last_wakeup_notification_time
                                ) > timedelta(minutes=5):
                                    # Calculate in-bed info for alert
                                    in_bed_info = {}
                                    if wakeup_require_in_bed and last_person_detected_time is not None:
                                        minutes_since = (now - last_person_detected_time).total_seconds() / 60
                                        in_bed_info = {
                                            "person_in_bed": person_was_in_bed,
                                            "last_detection_minutes_ago": round(minutes_since, 1),
                                            "detection_window_minutes": person_detection_window_minutes,
                                        }

                                    logger.info(
                                        f"Wake-up detected: avg motion {avg_motion:.2f} over {sleep_minutes}min, "
                                        f"current {current_motion:.2f}"
                                        + (f", person in bed: {person_was_in_bed}" if wakeup_require_in_bed else "")
                                    )

                                    # Track wake-up time for seizure sensitivity boost
                                    global last_wakeup_time
                                    last_wakeup_time = now

                                    wakeup_alert_data = {
                                        "timestamp": now.isoformat(),
                                        "type": "wakeup",
                                        "sleep_duration_minutes": sleep_minutes,
                                        "avg_motion_during_sleep": float(avg_motion),
                                        "current_motion": float(current_motion),
                                        "person_detection_enabled": wakeup_require_in_bed,
                                        **in_bed_info,  # Add in-bed info if available
                                        "is_test": False,
                                    }
                                    
                                    try:
                                        await notification_manager.send_alert(wakeup_alert_data)
                                        last_wakeup_notification_time = now
                                        logger.info("✓ Wake-up notification sent - seizure sensitivity boosted for next 15 minutes")
                                    except Exception as e:
                                        logger.error(f"Failed to send wake-up notification: {e}")
                
            except Exception as e:
                logger.error(f"Motion feature extraction error: {e}")
            
            # Only run expensive processing (pose detection, model inference) if needed
            if should_process_frame:
                last_processing_time = datetime.now()
                
                # Apply wake-triggered sensitivity boost if within post-wake window
                if last_wakeup_time is not None:
                    minutes_since_wake = (datetime.now(timezone.utc) - last_wakeup_time).total_seconds() / 60
                    
                    if minutes_since_wake <= POST_WAKE_SENSITIVITY_MINUTES:
                        # Within post-wake window: use lower threshold (more sensitive)
                        seizure_detector.set_pose_threshold(POST_WAKE_POSE_THRESHOLD)
                    else:
                        # Outside post-wake window: restore normal threshold
                        seizure_detector.set_pose_threshold(NORMAL_POSE_THRESHOLD)
                else:
                    # No wake-up detected yet: use normal threshold
                    seizure_detector.set_pose_threshold(NORMAL_POSE_THRESHOLD)
                
                    # Run multi-stage seizure detection pipeline
                    # Pipeline now handles pose extraction internally
                    # Returns both detection result and frame with pose skeleton drawn
                    # Use video-based timestamps when analyzing video (not wall-clock time)
                    if analyzing_video and video_start_timestamp is not None:
                        # Calculate timestamp based on video frame progression
                        # Assume 5 FPS for video files (standard for recorded seizure videos)
                        global video_frame_count
                        video_frame_count += 1
                        timestamp = video_start_timestamp + (video_frame_count / 5.0)
                    else:
                        timestamp = datetime.now()
                    
                    detection_result, drawn_frame = seizure_detector.analyze_frame(
                        frame, timestamp, show_overlay=show_overlay
                    )
                    # Use drawn frame if overlay is enabled
                    if show_overlay:
                        processed_frame = drawn_frame

                    # Get alert strategy from environment
                    alert_strategy = os.getenv("ALERT_STRATEGY", "onset_only")
                    should_send_alert = seizure_detector.should_alert(detection_result, strategy=alert_strategy)
                    
                    # Handle alerts with cooldown
                    if should_send_alert:
                        current_time = datetime.now()
                        if last_alert_time is None or (current_time - last_alert_time).total_seconds() > alert_cooldown_seconds:
                            # Determine alert type
                            alert_type = "onset" if detection_result.seizure_onset else "offset" if detection_result.seizure_offset else "update"
                            
                            # Create alert data from detection result
                            alert_data = detection_result.to_dict()
                            alert_data["alert_type"] = alert_type
                            
                            await handle_seizure_alert(alert_data)
                            last_alert_time = current_time
                            
                            logger.info(f"Alert sent: {alert_type}, duration: {detection_result.seizure_current_duration}s")
                        else:
                            logger.debug(f"Alert suppressed - cooldown active ({(current_time - last_alert_time).total_seconds():.1f}s elapsed)")
                    
                    # Handle seizure state exit when detector signals offset
                    if detection_result.seizure_offset:
                        global is_seizing, seizure_ended_at, current_seizure_id
                        
                        # Only update state if we were actually in seizure
                        if is_seizing or current_seizure_id:
                            is_seizing = False
                            seizure_ended_at = datetime.now()
                            
                            # Close seizure log entry (skip in test mode)
                            if current_seizure_id and not analyzing_video:
                                try:
                                    await seizure_logger.end(current_seizure_id)
                                    logger.info(f"✓ Closed seizure log entry: {current_seizure_id}")
                                except Exception as e:
                                    logger.error(f"Failed to close seizure log: {e}")
                            
                            # Stop recording if active
                            if video_recorder.is_recording():
                                try:
                                    video_recorder.stop_recording()
                                    logger.info("✓ Stopped video recording")
                                except Exception as e:
                                    logger.error(f"Failed to stop recording: {e}")
                            
                            # Clear seizure ID
                            current_seizure_id = None
                            
                            logger.warning("=" * 70)
                            duration_str = f"{detection_result.seizure_current_duration:.1f}s" if detection_result.seizure_current_duration else "unknown"
                            logger.warning(f"✓ SEIZURE ENDED (duration: {duration_str})")
                            logger.warning("=" * 70)

                # Add detection info to analysis result
                analysis_result["detection"] = detection_result.to_dict()
            
            # Only produce the live-preview frame when the dashboard is actively
            # polling /frame. Detection, recording, HR and alerting above run every
            # frame regardless; this just skips the throwaway overlay/JPEG/base64
            # encode when nobody is watching the stream (the common case overnight).
            stream_active = (
                last_frame_request_time is not None
                and (datetime.now() - last_frame_request_time).total_seconds()
                < STREAM_IDLE_SECONDS
            )
            if stream_active:
                # Add motion visualization overlay if enabled (draw on top of everything)
                if show_overlay and motion_features:
                    try:
                        motion_history = list(motion_extractor.motion_magnitude_history)
                        processed_frame = motion_visualizer.draw_motion_overlay(
                            processed_frame, motion_features, motion_history, mode='heatmap'
                        )
                    except Exception as e:
                        logger.error(f"Motion visualization error: {e}")

                # HR strip — independent of motion overlay.
                if show_overlay and motion_visualizer is not None and GarminHRConfig.ENABLED:
                    try:
                        processed_frame = motion_visualizer.draw_hr_overlay(
                            processed_frame, _build_hr_overlay_state()
                        )
                    except Exception as e:
                        logger.error(f"HR overlay error: {e}")

                # Encode frame
                encode_frame = processed_frame if show_overlay else frame
                _, buffer = cv2.imencode(
                    ".jpg",
                    encode_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, APIConfig.JPEG_QUALITY],
                )
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

                # Helper to recursively convert numpy types to Python native types for JSON serialization
                def sanitize_for_json(obj):
                    if obj is None:
                        return None
                    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                        return int(obj)
                    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                        return float(obj)
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                    if isinstance(obj, dict):
                        return {k: sanitize_for_json(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [sanitize_for_json(item) for item in obj]
                    return obj

                # Create message
                message = sanitize_for_json({
                    "frame": frame_base64,
                    "analysis": analysis_result,
                    "video_res": camera_manager.resolution,
                })

                # Update frame queue
                while frame_queue.full():
                    await frame_queue.get()
                await frame_queue.put(message)

            # Record if active
            if video_recorder.is_recording():
                video_recorder.write_frame(frame)

            # Log data periodically
            # Use max confidence from either model stage
            max_confidence = max(detection_result.motion_model_confidence, detection_result.pose_model_confidence)
            if motion_detected or max_confidence > 0.3:
                # Get current motion magnitude
                current_motion_mag = 0.0
                if motion_extractor and len(motion_extractor.motion_magnitude_history) > 0:
                    current_motion_mag = float(motion_extractor.motion_magnitude_history[-1])
                
                await data_logger.save(
                    {
                        "motion_magnitude": current_motion_mag,
                        "motion_degree": motion_degree,  # Keep for compatibility
                        "seizure_confidence": max_confidence,
                        "stage_reached": detection_result.stage_reached,
                        "confirmed": detection_result.pose_model_detected or detection_result.motion_model_detected,
                    }
                )

            # Adaptive sleep based on motion check rate
            # This ensures we check for motion frequently enough to detect wake-ups
            await asyncio.sleep(1 / motion_check_fps)

    except asyncio.CancelledError:
        logger.info("Video processing loop cancelled")
    except Exception as e:
        logger.exception(f"Error in video processing loop: {e}")
    finally:
        logger.info("Video processing loop ended")


@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    logger.info("Starting Seizure Detection Server v2.0")

    # Initialize pose detection backend (MediaPipe/Hailo)
    logger.info("Initializing pose detection...")
    setup_pose_detection()

    # Start camera
    logger.info("Starting camera on macOS...")
    camera_started = await camera_manager.start()
    if camera_started:
        logger.info("Camera started successfully")
    else:
        logger.error("Failed to start camera")

    # Start video processing
    asyncio.create_task(process_video_frames())

    # Start connectivity monitoring
    asyncio.create_task(
        connectivity_checker.monitor(
            is_recording_func=lambda: video_recorder.is_recording(),
            is_seizing_func=lambda: is_seizing,
        )
    )

    # Enable pose detection (MediaPipe/Hailo backend auto-detected)
    global use_pose_detection
    use_pose_detection = True

    # Initialize pose detection backend
    try:
        setup_pose_detection()
        logger.info("✓ Pose detection initialized (backend auto-detected)")
    except Exception as e:
        logger.error(f"Failed to initialize pose detection: {e}")
        use_pose_detection = False

    # Garmin heart-rate monitor (BLE). Runs in background; missing watch is non-fatal.
    if GarminHRConfig.ENABLED:
        global garmin_monitor, hr_detector, hr_detector_task
        try:
            from seizure_detection.garmin_hr import GarminHRMonitor
            from seizure_detection.analysis.hr_detector import HRDetector
            garmin_monitor = GarminHRMonitor(
                address=GarminHRConfig.ADDRESS,
                name_filter=GarminHRConfig.NAME_FILTER,
                reconnect_delay=GarminHRConfig.RECONNECT_DELAY_SEC,
                scan_timeout=GarminHRConfig.SCAN_TIMEOUT_SEC,
                rescan_interval=GarminHRConfig.RESCAN_INTERVAL_SEC,
                connect_timeout=GarminHRConfig.CONNECT_TIMEOUT_SEC,
                clear_bluez_on_retry=GarminHRConfig.CLEAR_BLUEZ_ON_RETRY,
                notification_stale_sec=GarminHRConfig.NOTIFICATION_STALE_SEC,
                disconnect_timeout=GarminHRConfig.DISCONNECT_TIMEOUT_SEC,
            )
            await garmin_monitor.start()
            hr_detector = HRDetector(
                garmin_monitor,
                on_alert=handle_hr_alert,
                hr_logger=hr_logger,
            )
            hr_detector_task = asyncio.create_task(
                hr_detector.run(), name="hr-detector"
            )
            logger.info("✓ Garmin HR monitor + detector started")
        except ImportError as e:
            logger.warning(
                f"Garmin HR enabled but `bleak` is not installed: {e}. "
                "Run: pip install bleak"
            )
            garmin_monitor = None
        except Exception as e:
            logger.warning(f"Garmin HR monitor failed to start: {e}")
            garmin_monitor = None

    logger.info("Server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down server...")

    # Stop recording
    if video_recorder.is_recording():
        await video_recorder.stop()

    # Stop camera
    if camera_manager.is_running():
        await camera_manager.stop()

    # Release pose detection (only once, with error handling)
    if use_pose_detection:
        try:
            release_pose_detection()
        except Exception as e:
            logger.warning(f"Error releasing pose detection: {e}")

    # Stop HR detector + Garmin HR monitor
    if hr_detector_task is not None:
        hr_detector_task.cancel()
        try:
            await hr_detector_task
        except (asyncio.CancelledError, Exception):
            pass
    if garmin_monitor is not None:
        try:
            await garmin_monitor.stop()
            logger.info("✓ Garmin HR monitor stopped")
        except Exception as e:
            logger.warning(f"Error stopping Garmin HR monitor: {e}")

    logger.info("Server shutdown complete")


if __name__ == "__main__":
    import uvicorn

    # Configure uvicorn to suppress access logs
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=log_config,
        access_log=False,  # Disable access logs entirely
    )
