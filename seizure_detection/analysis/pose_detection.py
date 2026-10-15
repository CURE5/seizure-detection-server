import time
import logging

import cv2
import numpy as np
from collections import deque
from types import SimpleNamespace

from .one_euro_filter import OneEuroFilter
from ..config import FORCE_MEDIAPIPE, SeizureDetectionConfig

logger = logging.getLogger(__name__)

# Detect which pose tracking backend is available
# Priority: FORCE_MEDIAPIPE flag > Hailo AI (RPi 5 hardware accelerated) > MediaPipe > None
BACKEND = "none"
is_pose_detection_available = False
hailo_detector = None

# Frame counter for monotonically increasing timestamps (MediaPipe requirement)
_frame_counter = 0
_last_processed_timestamp = 0  # Track when async callbacks last fired

# Check if MediaPipe is forced (required for hypermotor_detector.onnx)
if FORCE_MEDIAPIPE or SeizureDetectionConfig.USE_CUSTOM_MODEL:
    logger.info("🔧 MediaPipe forced via FORCE_MEDIAPIPE or custom model usage")
    # Skip Hailo detection, go straight to MediaPipe
    SKIP_HAILO = True
else:
    SKIP_HAILO = False

# Try Hailo AI first (hardware accelerated on RPi 5) - unless MediaPipe is forced
if not SKIP_HAILO:
    try:
        from .hailo_pose import HAILO_AVAILABLE

        if HAILO_AVAILABLE:
            BACKEND = "hailo"
            is_pose_detection_available = True
            logger.info("✓ Hailo AI backend available for pose detection")
            logger.info("  Hardware: Hailo-8L AI accelerator")
            logger.info("  Model: YOLOv8-Pose (17 COCO keypoints)")
        else:
            raise ImportError("Hailo platform not available")
    except Exception as e:
        logger.debug(f"Hailo AI not available: {e}")
        SKIP_HAILO = True
else:
    logger.info("⏩ Skipping Hailo detection (MediaPipe forced)")

# Fallback to MediaPipe if Hailo not used
if SKIP_HAILO or BACKEND == "none":
    # Try MediaPipe
    try:
        import mediapipe as mp
        from .gaze_detection import gaze, gaze_direction
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.framework.formats import landmark_pb2
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
        from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark

        mp_pose = mp.solutions.pose
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        BACKEND = "mediapipe"
        is_pose_detection_available = True
        logger.info("✓ MediaPipe backend for pose detection")
        logger.info("  Proven: 100% recall in training data")
    except Exception as e2:
        logger.warning(f"MediaPipe not available: {e2}")
        mp = None
        python = None
        vision = None
        landmark_pb2 = None
        PoseLandmarker = None
        FaceLandmarker = None
        NormalizedLandmark = None
        mp_pose = mp_face_mesh = mp_drawing = mp_drawing_styles = None
        gaze = gaze_direction = None
        BACKEND = "none"
        is_pose_detection_available = False
        logger.warning("⚠ No pose detection backend available")

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()

# Initialize the pose landmarker model
pose_detector = None
pose_results = SimpleNamespace()
pose_results.raw = None
pose_results.smoothed = None
face_detector = None
face_results = None

# Configuration based on backend
if BACKEND == "hailo":
    # Hailo AI configuration
    pose_model = "./hailo_models/yolov8_pose.hef"  # Will need to be configured
    num_poses = 4
    min_pose_detection_confidence = 0.5
elif BACKEND == "mediapipe":
    # MediaPipe configuration
    pose_model = "./mediapipe_models/pose_landmarker.task"
    face_model = "./mediapipe_models/face_landmarker.task"
    num_poses = 4
    min_pose_detection_confidence = 0.8
    min_pose_presence_confidence = 0.5
    min_tracking_confidence = 0.5
    output_segmentation_masks = True

    running_mode = vision.RunningMode.LIVE_STREAM
    min_face_presence_confidence = 0.05
    min_face_detection_confidence = 0.05
    num_faces = 1
    output_face_blendshapes = False  # Whether Face Landmarker outputs face blendshapes
    
    # One Euro Filter configuration for landmark smoothing
    # min_cutoff: baseline smoothing for low-speed movements (lower = heavier smoothing, less jitter)
    # beta: speed-based smoothing factor (higher = more responsive to rapid movements)
    min_cutoff = 0.7
    beta = 0.01
    
    # Initialize filters for each landmark's (x, y, z) coordinates (33 landmarks in MediaPipe Pose)
    landmark_filters = [
        [OneEuroFilter(min_cutoff=min_cutoff, beta=beta),  # Filter for x
         OneEuroFilter(min_cutoff=min_cutoff, beta=beta),  # Filter for y
         OneEuroFilter(min_cutoff=min_cutoff, beta=beta)]  # Filter for z
        for _ in range(33)  # 33 pose landmarks
    ]
    
    # Smoothing buffer for temporal smoothing (stores last N frames)
    smoothing_buffer = deque(maxlen=5)
    previous_landmarks = None  # For low-pass filter
    alpha = 0.8  # Low-pass filter smoothing factor (higher = smoother)
else:
    # Default values if no backend
    landmark_filters = []
    smoothing_buffer = deque(maxlen=5)
    previous_landmarks = None
    alpha = 0.8


def extract_landmarks(pose_landmarks):
    """Convert landmarks to a 2D numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks])


def smooth_landmarks(landmarks):
    """Apply moving average smoothing over recent frames."""
    if BACKEND != "mediapipe":
        return landmarks
    
    # Add current landmarks to the buffer
    smoothing_buffer.append(np.array(landmarks))
    
    # Calculate the moving average over the buffer
    smoothed = np.mean(smoothing_buffer, axis=0)
    return smoothed


def low_pass_filter(landmarks):
    """Apply low-pass filter to reduce jitter."""
    global previous_landmarks
    
    if BACKEND != "mediapipe":
        return landmarks
    
    if previous_landmarks is None:
        previous_landmarks = landmarks
        return landmarks
    
    # Apply low-pass filter
    smoothed = [
        alpha * prev + (1 - alpha) * curr
        for prev, curr in zip(previous_landmarks, landmarks)
    ]
    previous_landmarks = smoothed
    return smoothed


def filter_landmarks_one_euro(landmarks, filters, timestamp):
    """
    Apply One Euro Filter to a set of landmarks.
    
    Parameters:
        landmarks: List of landmarks [(x, y, z), ...].
        filters: List of One Euro Filter instances for each (x, y, z) coordinate.
        timestamp: Current time in seconds.
    
    Returns:
        Smoothed landmarks as a list of (x, y, z).
    """
    if BACKEND != "mediapipe":
        return landmarks
    
    smoothed_landmarks = []
    for idx, (lm, lm_filters) in enumerate(zip(landmarks, filters)):
        x_filtered = lm_filters[0].filter(lm[0], timestamp)
        y_filtered = lm_filters[1].filter(lm[1], timestamp)
        z_filtered = lm_filters[2].filter(lm[2], timestamp)
        smoothed_landmarks.append([x_filtered, y_filtered, z_filtered])
    return smoothed_landmarks


def convert_to_normalized_landmarks(smoothed_landmarks):
    """
    Convert a list of smoothed landmarks (x, y, z) back to NormalizedLandmark objects.
    
    Parameters:
        smoothed_landmarks: List of [x, y, z] landmarks.
    
    Returns:
        List of NormalizedLandmark objects.
    """
    if BACKEND != "mediapipe":
        return smoothed_landmarks
    
    normalized_landmarks = []
    for lm in smoothed_landmarks:
        normalized_landmarks.append(
            NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2])
        )
    return normalized_landmarks


def extract_eyes_and_hands(pose_landmarks_list):
    """Extract specific landmarks for eyes and hands tracking."""
    if not pose_landmarks_list:
        return None
    
    all_landmarks = []
    
    # Iterate through detected persons
    for person_landmarks in pose_landmarks_list:
        specific_landmarks = {
            "LEFT_EYE": {
                "x": person_landmarks[1].x,
                "y": person_landmarks[1].y,
                "z": person_landmarks[1].z,
            },
            "RIGHT_EYE": {
                "x": person_landmarks[2].x,
                "y": person_landmarks[2].y,
                "z": person_landmarks[2].z,
            },
            "LEFT_WRIST": {
                "x": person_landmarks[15].x,
                "y": person_landmarks[15].y,
                "z": person_landmarks[15].z,
            },
            "RIGHT_WRIST": {
                "x": person_landmarks[16].x,
                "y": person_landmarks[16].y,
                "z": person_landmarks[16].z,
            },
        }
        all_landmarks.append(specific_landmarks)
    
    return all_landmarks


def setup():
    """Initialize the pose detection backend (Hailo, MediaPipe, or no-op)."""
    global pose_detector, face_detector, hailo_detector

    if BACKEND == "none":
        logger.info("setup(): No pose detection backend available — no-op")
        return

    elif BACKEND == "hailo":
        logger.info("Setting up Hailo AI pose detection...")
        try:
            from .hailo_pose import HailoPoseDetector

            hailo_detector = HailoPoseDetector()
            logger.info("✓ Hailo AI pose detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            raise

    elif BACKEND == "mediapipe":
        logger.info("Setting up MediaPipe pose detection...")

        # PoseLandmarker callback
        def pose_callback(
            result: vision.PoseLandmarkerResult,
            unused_output_image: mp.Image,
            timestamp_ms: int,
        ):
            global pose_results, _last_processed_timestamp
            # Track that callback was actually invoked
            _last_processed_timestamp = timestamp_ms
            
            if result:
                pose_results = SimpleNamespace()
                pose_results.raw = result
                pose_results.smoothed = result
            else:
                pose_results.raw = None
                pose_results.smoothed = None

        # FaceLandmarker callback
        def face_callback(
            result: vision.FaceLandmarkerResult,
            unused_output_image: mp.Image,
            timestamp_ms: int,
        ):
            global face_results
            if result:
                face_results = result
            else:
                face_results = None

        # Initialize the pose landmarker model
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model),
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
            result_callback=pose_callback,
        )
        pose_detector = vision.PoseLandmarker.create_from_options(options)

        face_options = FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_model),
            running_mode=running_mode,
            min_face_presence_confidence=min_face_presence_confidence,
            min_face_detection_confidence=min_face_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_faces=num_faces,
            output_face_blendshapes=output_face_blendshapes,
            result_callback=face_callback,
        )
        face_detector = FaceLandmarker.create_from_options(face_options)
        logger.info("✓ MediaPipe backend initialized")


def release():
    """Release resources for the active pose detection backend."""
    global pose_detector, face_detector

    if BACKEND == "none":
        logger.info("release(): No backend to release")
        return

    elif BACKEND == "hailo":
        logger.info("Releasing Hailo AI resources...")
        if pose_detector:
            try:
                # Hailo PoseEstimation cleanup
                pose_detector.release()
                pose_detector = None
            except Exception as e:
                logger.warning(f"Error releasing Hailo resources: {e}")
        logger.info("✓ Hailo AI resources released")

    elif BACKEND == "mediapipe":
        logger.info("Releasing MediaPipe resources...")
        try:
            if pose_detector:
                pose_detector.close()
        except Exception as e:
            logger.warning(f"Error closing pose detector: {e}")
        try:
            if face_detector:
                face_detector.close()
        except Exception as e:
            logger.warning(f"Error closing face detector: {e}")
        logger.info("✓ MediaPipe resources released")


# Function to extract specific landmarks
def extract_eyes_and_hands(pose_landmarks_list):
    if not pose_landmarks_list:
        return None

    all_landmarks = []

    # Iterate through detected persons
    for person_landmarks in pose_landmarks_list:
        specific_landmarks = {
            "LEFT_EYE": {
                "x": person_landmarks[1].x,
                "y": person_landmarks[1].y,
                "z": person_landmarks[1].z,
            },
            "RIGHT_EYE": {
                "x": person_landmarks[2].x,
                "y": person_landmarks[2].y,
                "z": person_landmarks[2].z,
            },
            "LEFT_WRIST": {
                "x": person_landmarks[15].x,
                "y": person_landmarks[15].y,
                "z": person_landmarks[15].z,
            },
            "RIGHT_WRIST": {
                "x": person_landmarks[16].x,
                "y": person_landmarks[16].y,
                "z": person_landmarks[16].z,
            },
        }
        all_landmarks.append(specific_landmarks)

    return all_landmarks


# Buffer to store past frames for smoothing
smoothing_buffer = deque(maxlen=5)  # Store the last 5 frames


def smooth_landmarks(landmarks):
    # Add current landmarks to the buffer
    smoothing_buffer.append(np.array(landmarks))

    # Calculate the moving average over the buffer
    smoothed = np.mean(smoothing_buffer, axis=0)
    return smoothed


def extract_landmarks(pose_landmarks):
    # Convert landmarks to a 2D array
    return np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks])


alpha = 0.8  # Smoothing factor (higher = smoother)

# Initialize previous landmarks for filtering
previous_landmarks = None


def low_pass_filter(landmarks):
    global previous_landmarks

    if previous_landmarks is None:
        previous_landmarks = landmarks
        return landmarks

    # Apply low-pass filter
    smoothed = [
        alpha * prev + (1 - alpha) * curr
        for prev, curr in zip(previous_landmarks, landmarks)
    ]
    previous_landmarks = smoothed
    return smoothed


# Initialize a filter for each landmark's (x, y, z) coordinates
min_cutoff = 0.7  # min_cutoff determines the baseline smoothing for low-speed movements. A lower value results in heavier smoothing and less jitter.
beta = 0.01  # Speed-based smoothing factor Higher beta: Makes the filter more responsive to rapid movements but may increase jitter.
landmark_filters = [
    [
        OneEuroFilter(min_cutoff=min_cutoff, beta=beta),  # Filter for x
        OneEuroFilter(min_cutoff=min_cutoff, beta=beta),  # Filter for y
        OneEuroFilter(min_cutoff=min_cutoff, beta=beta),
    ]  # Filter for z
    for _ in range(33)  # Adjust for the number of landmarks
]


def filter_landmarks(landmarks, filters, timestamp):
    """
    Apply One Euro Filter to a set of landmarks.

    Parameters:
        landmarks: List of landmarks [(x, y, z), ...].
        filters: List of One Euro Filter instances for each (x, y, z) coordinate.
        timestamp: Current time in seconds.

    Returns:
        Smoothed landmarks as a list of (x, y, z).
    """
    smoothed_landmarks = []
    for idx, (lm, lm_filters) in enumerate(zip(landmarks, filters)):
        x_filtered = lm_filters[0].filter(lm[0], timestamp)
        y_filtered = lm_filters[1].filter(lm[1], timestamp)
        z_filtered = lm_filters[2].filter(lm[2], timestamp)
        smoothed_landmarks.append([x_filtered, y_filtered, z_filtered])
    return smoothed_landmarks


def convert_to_normalized_landmarks(smoothed_landmarks):
    """
    Convert a list of smoothed landmarks (x, y, z) back to a list of NormalizedLandmark objects.

    Parameters:
        smoothed_landmarks: List of [x, y, z] landmarks.

    Returns:
        List of NormalizedLandmark objects.
    """
    normalized_landmarks = []
    for lm in smoothed_landmarks:
        normalized_landmarks.append(NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2]))
    return normalized_landmarks


def serialize_pose_landmarker_result(result):
    if result is None:
        return {}

    # Extract specific tracking points (eyes and hands) for additional analysis
    specific_landmarks = extract_eyes_and_hands(result.pose_landmarks)
    
    # Also return full landmarks for model input
    all_landmarks = []
    for landmark_list in result.pose_landmarks:
        landmarks = [
            {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': getattr(landmark, 'visibility', 1.0)
            }
            for landmark in landmark_list
        ]
        all_landmarks.append(landmarks)
    
    return {
        'landmarks': all_landmarks,
        'specific': specific_landmarks  # Eyes and hands for tracking
    }


def inv_rot90_cw_point(xr, yr, W, H):
    """
    Inverse rotation for 90° clockwise rotation.
    Maps point (xr, yr) in rotated frame back to (x, y) in original frame.
    
    Args:
        xr, yr: Coordinates in rotated frame
        W, H: Original frame width and height
    
    Returns:
        (x, y): Coordinates in original frame
    """
    x = W - 1 - yr
    y = xr
    return int(x), int(y)


def inv_rot90_cw_bbox(xr, yr, wr, hr, W, H):
    """
    Inverse rotation for bounding box after 90° clockwise rotation.
    Transforms bbox by transforming its corners.
    
    Args:
        xr, yr, wr, hr: Bbox (x, y, width, height) in rotated frame
        W, H: Original frame width and height
    
    Returns:
        (x, y, w, h): Bbox in original frame coordinates
    """
    # Transform bbox corners
    x1r, y1r = xr, yr
    x2r, y2r = xr + wr - 1, yr + hr - 1
    
    x1, y1 = inv_rot90_cw_point(x1r, y1r, W, H)
    x2, y2 = inv_rot90_cw_point(x2r, y2r, W, H)
    
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def process_frame(frame, do_draw=True, do_flip=False):
    """Process a video frame with the active pose detection backend."""
    global pose_detector, face_detector, pose_results, face_results

    # No backend available - return frame unchanged
    if BACKEND == "none":
        return frame, {"gaze": None, "pose": {}}

    if do_flip:
        image = cv2.flip(frame, 1)
    else:
        image = frame

    # Backend-specific processing
    if BACKEND == "hailo":
        # ═══════════════════════════════════════════════════════════════════
        # HAILO AI POSE PROCESSING PIPELINE
        # ═══════════════════════════════════════════════════════════════════
        # This processes each video frame through the Hailo-8L AI accelerator
        # to detect human poses for seizure detection analysis.
        #
        # PIPELINE FLOW:
        # 1. Frame → Hailo Hardware → 17 COCO Keypoints
        # 2. Keypoints → Normalized Landmarks (0-1 range)
        # 3. Landmarks → SeizureDetector.add_pose_data()
        # 4. SeizureDetector → Analyzes motion patterns
        #
        # SEIZURE DETECTION STRATEGY:
        # ═══════════════════════════════════════════════════════════════════
        # The keypoints from Hailo feed into SeizureDetector which monitors:
        #
        # 1. RHYTHMIC MOTION (2-5 Hz oscillations)
        #    - Tracks wrist positions (keypoints 9, 10) over time
        #    - Detects repetitive shaking or jerking
        #    - Common in tonic-clonic seizures
        #
        # 2. VELOCITY SPIKES (sudden movement changes)
        #    - Calculates frame-to-frame displacement of joints
        #    - Detects rapid, uncontrolled movements
        #    - Indicates loss of motor control
        #
        # 3. ACCELERATION PATTERNS (sharp jerks)
        #    - Second derivative of position (velocity change rate)
        #    - Identifies sudden starts/stops of movement
        #    - Characteristic of myoclonic seizures
        #
        # 4. POSTURE CHANGES (body position shifts)
        #    - Monitors hip/shoulder alignment (keypoints 5,6,11,12)
        #    - Detects falling or collapse
        #    - Alerts for safety concerns
        #
        # KEY LANDMARKS FOR SEIZURE DETECTION:
        # - Wrists (9,10): Primary indicators of rhythmic shaking
        # - Shoulders (5,6): Upper body stability/tremor
        # - Hips (11,12): Core stability, falling detection
        # - Ankles (15,16): Lower body convulsions
        # ═══════════════════════════════════════════════════════════════════

        try:
            # Rotate frame 90° clockwise to present horizontal baby (head right) as upright
            # YOLOv8-pose is trained on COCO dataset with vertical/standing poses
            orig_h, orig_w = image.shape[:2]
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rot_h, rot_w = rotated.shape[:2]  # After rotation: rot_h=orig_w, rot_w=orig_h
            
            logger.info(f"DEBUG: Rotated frame {orig_w}×{orig_h} → {rot_w}×{rot_h}")
            
            # Run detection on rotated frame
            detections = hailo_detector.detect(rotated)
            
            # DEBUG: Log what we got from Hailo
            logger.info(f"Hailo detector returned: {type(detections)}, length: {len(detections) if detections else 0}")
            
            # Transform all detections back to original frame coordinates
            if detections and len(detections) > 0:
                for det in detections:
                    # Transform bbox using corner-based method
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    bbox_x, bbox_y, bbox_w, bbox_h = bbox
                    det["bbox"] = inv_rot90_cw_bbox(bbox_x, bbox_y, bbox_w, bbox_h, orig_w, orig_h)
                    
                    # Transform all keypoints
                    for kp in det.get("keypoints", []):
                        kp_x_rot = kp["x"]
                        kp_y_rot = kp["y"]
                        kp["x"], kp["y"] = inv_rot90_cw_point(kp_x_rot, kp_y_rot, orig_w, orig_h)
                
                logger.info(f"DEBUG: Transformed {len(detections)} detections back to original coords")

            # Convert Hailo detections to our standardized format
            pose_data = {}
            if detections and len(detections) > 0:
                # Filter detections: Only keep those with valid bboxes
                # and keypoints not stuck in letterbox padding
                valid_detections = []
                for idx, det in enumerate(detections):
                    bbox = det.get("bbox", (0, 0, 0, 0))
                    keypoints = det.get("keypoints", [])
                    
                    # Skip detections with invalid bounding boxes
                    # (negative coords or zero size means bad detection)
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    
                    if not keypoints:
                        continue
                    
                    # Check if keypoints span vertically (not all at y=0)
                    y_values = [kp["y"] for kp in keypoints if kp["confidence"] > 0.5]
                    if not y_values:
                        continue
                        
                    min_y = min(y_values)
                    max_y = max(y_values)
                    avg_y = sum(y_values) / len(y_values)
                    y_span = max_y - min_y
                    
                    # Valid pose should span vertically and not be at y=0
                    # Relaxed thresholds: 10% span (36px) and avg above 5% (18px)
                    if y_span > image.shape[0] * 0.1 and avg_y > image.shape[0] * 0.05:
                        valid_detections.append(det)
                        if idx < 3:  # Log first few for debugging
                            logger.info(
                                f"DEBUG: Det {idx} valid - bbox={bbox}, "
                                f"y_span={y_span:.0f}, avg_y={avg_y:.0f}"
                            )
                
                logger.info(
                    f"DEBUG: Filtered {len(detections)} detections "
                    f"→ {len(valid_detections)} valid"
                )
                
                if not valid_detections:
                    logger.warning("No valid detections found after filtering")
                    return image, {"gaze": None, "pose": {}, "backend": "hailo"}
                
                # Prefer detections that are larger (closer/more prominent)
                # Sort by bbox area (width * height)
                valid_detections.sort(
                    key=lambda d: d["bbox"][2] * d["bbox"][3], 
                    reverse=True
                )
                detection = valid_detections[0]
                
                logger.info(
                    f"DEBUG: Selected largest detection: "
                    f"area={detection['bbox'][2] * detection['bbox'][3]}"
                )
                
                # DEBUG: Log all keypoints of selected detection
                logger.info(f"DEBUG: Selected detection bbox={detection['bbox']}, conf={detection['confidence']:.3f}")
                for idx, kp in enumerate(detection.get("keypoints", [])[:5]):
                    logger.info(f"DEBUG: COCO kp{idx}: x={kp['x']}, y={kp['y']}, conf={kp['confidence']:.3f}")

                # Extract COCO keypoints from Hailo
                # COCO format (17 points): nose, left_eye, right_eye, left_ear, right_ear,
                # left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
                # left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
                coco_keypoints = detection.get("keypoints", [])

                # Map COCO indices to MediaPipe indices for ONNX model compatibility
                # ONNX was trained on MediaPipe's 33-point format
                coco_to_mediapipe = [
                    0,  # COCO 0 (nose) -> MediaPipe 0 (NOSE)
                    2,  # COCO 1 (left_eye) -> MediaPipe 2 (LEFT_EYE)
                    5,  # COCO 2 (right_eye) -> MediaPipe 5 (RIGHT_EYE)
                    7,  # COCO 3 (left_ear) -> MediaPipe 7 (LEFT_EAR)
                    8,  # COCO 4 (right_ear) -> MediaPipe 8 (RIGHT_EAR)
                    11,  # COCO 5 (left_shoulder) -> MediaPipe 11 (LEFT_SHOULDER)
                    12,  # COCO 6 (right_shoulder) -> MediaPipe 12 (RIGHT_SHOULDER)
                    13,  # COCO 7 (left_elbow) -> MediaPipe 13 (LEFT_ELBOW)
                    14,  # COCO 8 (right_elbow) -> MediaPipe 14 (RIGHT_ELBOW)
                    15,  # COCO 9 (left_wrist) -> MediaPipe 15 (LEFT_WRIST)
                    16,  # COCO 10 (right_wrist) -> MediaPipe 16 (RIGHT_WRIST)
                    23,  # COCO 11 (left_hip) -> MediaPipe 23 (LEFT_HIP)
                    24,  # COCO 12 (right_hip) -> MediaPipe 24 (RIGHT_HIP)
                    25,  # COCO 13 (left_knee) -> MediaPipe 25 (LEFT_KNEE)
                    26,  # COCO 14 (right_knee) -> MediaPipe 26 (RIGHT_KNEE)
                    27,  # COCO 15 (left_ankle) -> MediaPipe 27 (LEFT_ANKLE)
                    28,  # COCO 16 (right_ankle) -> MediaPipe 28 (RIGHT_ANKLE)
                ]

                # Create MediaPipe-compatible landmark array (33 points, filled with zeros for missing)
                # Initialize all 33 MediaPipe landmarks with low confidence
                pose_landmarks = []
                for i in range(33):
                    pose_landmarks.append(
                        {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "visibility": 0.0,  # Low visibility for missing points
                        }
                    )

                # Fill in the 17 COCO keypoints at their correct MediaPipe positions
                # DEBUG: Log image shape and first keypoint
                if coco_keypoints:
                    logger.info(f"DEBUG: Frame {orig_w}×{orig_h}, First kp: x={coco_keypoints[0]['x']}, y={coco_keypoints[0]['y']}")
                
                for coco_idx, kp in enumerate(coco_keypoints):
                    mp_idx = coco_to_mediapipe[coco_idx]

                    # Clamp coordinates to frame bounds
                    px = max(0, min(kp["x"], orig_w - 1))
                    py = max(0, min(kp["y"], orig_h - 1))
                    
                    # Normalize to 0-1 range
                    norm_x = px / orig_w
                    norm_y = py / orig_h
                    
                    # DEBUG: Log first normalization
                    if coco_idx == 0:
                        logger.info(f"DEBUG: Normalized: x={norm_x:.3f}, y={norm_y:.3f}")

                    pose_landmarks[mp_idx] = {
                        "x": norm_x,
                        "y": norm_y,
                        "z": 0.0,  # Hailo provides 2D only
                        "visibility": kp["confidence"],
                    }

                # Package in expected format for SeizureDetector
                # Extract specific key landmarks (eyes and wrists) for seizure detection
                specific_landmarks = {
                    "LEFT_EYE": {
                        "x": pose_landmarks[2]["x"],
                        "y": pose_landmarks[2]["y"],
                        "z": pose_landmarks[2]["z"],
                    },
                    "RIGHT_EYE": {
                        "x": pose_landmarks[5]["x"],
                        "y": pose_landmarks[5]["y"],
                        "z": pose_landmarks[5]["z"],
                    },
                    "LEFT_WRIST": {
                        "x": pose_landmarks[15]["x"],
                        "y": pose_landmarks[15]["y"],
                        "z": pose_landmarks[15]["z"],
                    },
                    "RIGHT_WRIST": {
                        "x": pose_landmarks[16]["x"],
                        "y": pose_landmarks[16]["y"],
                        "z": pose_landmarks[16]["z"],
                    },
                }
                pose_data = {
                    "pose_landmarks": [specific_landmarks]
                }

                # Optional visualization for debugging/monitoring
                logger.info(f"DEBUG: do_draw={do_draw}, valid_detections={len(valid_detections)}")
                
                # Draw ALL valid detections (not just selected one)
                if do_draw and len(valid_detections) > 0:
                    for det_idx, det in enumerate(valid_detections):
                        # Convert detection keypoints to MediaPipe format for drawing
                        det_coco_kpts = det.get("keypoints", [])
                        det_pose_landmarks = []
                        for i in range(33):
                            det_pose_landmarks.append({
                                "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0
                            })
                        
                        for coco_idx, kp in enumerate(det_coco_kpts):
                            mp_idx = coco_to_mediapipe[coco_idx]
                            det_pose_landmarks[mp_idx] = {
                                "x": kp["x"] / image.shape[1],
                                "y": kp["y"] / image.shape[0],
                                "z": 0.0,
                                "visibility": kp["confidence"],
                            }
                        
                        # Use different color schemes for selected vs other detections
                        is_selected = (det_idx == 0)  # First one is selected
                        
                        # Draw skeleton connections
                        pose_connections = [
                            # Face
                            (0, 1), (0, 4), (1, 2), (2, 3), (3, 7),
                            (4, 5), (5, 6), (6, 8),
                            # Torso
                            (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
                            # Left arm
                            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                            # Right arm
                            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                            # Left leg
                            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                            # Right leg
                            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
                        ]
                        
                        for connection in pose_connections:
                            start_idx, end_idx = connection
                            start_lm = det_pose_landmarks[start_idx]
                            end_lm = det_pose_landmarks[end_idx]
                            
                            if start_lm["visibility"] > 0.5 and end_lm["visibility"] > 0.5:
                                start_x = int(start_lm["x"] * image.shape[1])
                                start_y = int(start_lm["y"] * image.shape[0])
                                end_x = int(end_lm["x"] * image.shape[1])
                                end_y = int(end_lm["y"] * image.shape[0])
                                
                                # Selected: bright cyan, Others: dim gray
                                line_color = (255, 255, 0) if is_selected else (100, 100, 100)
                                line_thickness = 2 if is_selected else 1
                                cv2.line(image, (start_x, start_y), (end_x, end_y),
                                        line_color, line_thickness, cv2.LINE_AA)
                        
                        # Draw keypoints
                        for idx, landmark in enumerate(det_pose_landmarks):
                            if landmark["visibility"] > 0.2:
                                x = int(landmark["x"] * image.shape[1])
                                y = int(landmark["y"] * image.shape[0])
                                
                                if is_selected:
                                    # Selected detection: color-coded by importance
                                    if idx in [15, 16]:  # Wrists
                                        color = (0, 0, 255)  # Red
                                        radius = 6
                                    elif idx in [27, 28]:  # Ankles
                                        color = (255, 0, 0)  # Blue
                                        radius = 6
                                    elif idx in [11, 12, 23, 24]:  # Torso
                                        color = (0, 255, 255)  # Yellow
                                        radius = 5
                                    else:
                                        color = (0, 255, 0)  # Green
                                        radius = 4
                                else:
                                    # Other detections: dim white circles
                                    color = (150, 150, 150)
                                    radius = 3
                                
                                cv2.circle(image, (x, y), radius, color, -1)
                                cv2.circle(image, (x, y), radius, (0, 0, 0), 1)
                        
                        # Draw bounding box and label
                        bbox = det["bbox"]
                        bbox_color = (0, 255, 255) if is_selected else (80, 80, 80)
                        bbox_thickness = 2 if is_selected else 1
                        cv2.rectangle(image, 
                                    (bbox[0], bbox[1]),
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                    bbox_color, bbox_thickness)
                        
                        # Label with detection number and confidence
                        label = f"#{det_idx+1}" if is_selected else f"#{det_idx+1}"
                        label += f" {det['confidence']:.2f}"
                        cv2.putText(image, label,
                                  (bbox[0], bbox[1] - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, bbox_color, 1, cv2.LINE_AA)
                    
                    logger.info(f"DEBUG: Drew {len(valid_detections)} detections")

            # TEST: Draw test shapes to verify cv2 drawing is working
            # if do_draw:
   
            #     height, width = image.shape[:2]
            #     # Add text showing detection count
            #     cv2.putText(
            #         image,
            #         f"Detections: {len(detections) if detections else 0}",
            #         (10, height - 20),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.7,
            #         (255, 255, 255),
            #         2,
            #     )

            # Return processed frame and pose data
            # Note: Hailo doesn't provide gaze/face landmarks (pose-only model)
            return image, {
                "gaze": None,  # Not available from Hailo pose model
                "pose": pose_data,  # → Goes to SeizureDetector.add_pose_data()
                "backend": "hailo",
            }

        except Exception as e:
            logger.error(f"Hailo processing error: {e}")
            return image, {"gaze": None, "pose": {}, "backend": "hailo"}

    elif BACKEND == "mediapipe":
        # MediaPipe processing
        height, width, _ = frame.shape
        size = max(height, width)
        frame = cv2.copyMakeBorder(
            frame, 0, size - height, 0, size - width, cv2.BORDER_CONSTANT
        )

        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Use monotonically increasing frame counter for timestamps
        global _frame_counter, _last_processed_timestamp
        _frame_counter += 1
        timestamp_ms = _frame_counter
        
        # Check if we're stuck processing same timestamp (indicates async callback hung)
        if timestamp_ms - _last_processed_timestamp > 30:  # 30 frames behind = ~1 second at 30fps
            logger.warning(f"MediaPipe async callbacks may be stalled (current={timestamp_ms}, last_processed={_last_processed_timestamp})")
            # Don't block - continue with stale results rather than hanging

        # Run PoseLandmarker asynchronously
        try:
            pose_detector.detect_async(mp_image, timestamp_ms)
        except Exception as e:
            logger.error(f"PoseLandmarker error: {e}")

        # Run FaceLandmarker asynchronously
        try:
            face_detector.detect_async(mp_image, timestamp_ms)
        except Exception as e:
            logger.error(f"FaceLandmarker error: {e}")

        current_frame = image
        gaze_orientation = None

        # Draw pose landmarks if available
        if pose_results.smoothed and do_draw:
            for pose_landmarks in pose_results.smoothed.pose_landmarks:
                # Draw the pose landmarks
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in pose_landmarks
                    ]
                )
                mp_drawing.draw_landmarks(
                    current_frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style(),
                )

        # Process face landmarks and gaze if available
        if face_results:
            for face_landmarks in face_results.face_landmarks:
                # Draw the face landmarks.
                if do_draw and False:
                    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    face_landmarks_proto.landmark.extend(
                        [
                            landmark_pb2.NormalizedLandmark(
                                x=landmark.x, y=landmark.y, z=landmark.z
                            )
                            for landmark in face_landmarks
                        ]
                    )
                    mp_drawing.draw_landmarks(
                        image=current_frame,
                        landmark_list=face_landmarks_proto,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=current_frame,
                        landmark_list=face_landmarks_proto,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=current_frame,
                        landmark_list=face_landmarks_proto,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )

                # Process gaze for this face
                left_eye_gaze, right_eye_gaze = gaze(
                    current_frame, face_landmarks, do_draw=do_draw
                )
                gaze_orientation = gaze_direction(
                    current_frame, face_landmarks, do_draw=do_draw
                )

        # Return results for MediaPipe backend
        return current_frame, {
            "gaze": gaze_orientation,
            "pose": serialize_pose_landmarker_result(pose_results.smoothed),
            "backend": "mediapipe",
        }

    # Fallback return (should not reach here)
    return frame, {"gaze": None, "pose": {}, "backend": "unknown"}


# def run(model: str, num_poses: int,
#         min_pose_detection_confidence: float,
#         min_pose_presence_confidence: float, min_tracking_confidence: float,
#         output_segmentation_masks: bool,
#         camera_id: int, width: int, height: int) -> None:
#     """Continuously run inference on images acquired from the camera.

#   Args:
#       model: Name of the pose landmarker model bundle.
#       num_poses: Max number of poses that can be detected by the landmarker.
#       min_pose_detection_confidence: The minimum confidence score for pose
#         detection to be considered successful.
#       min_pose_presence_confidence: The minimum confidence score of pose
#         presence score in the pose landmark detection.
#       min_tracking_confidence: The minimum confidence score for the pose
#         tracking to be considered successful.
#       output_segmentation_masks: Choose whether to visualize the segmentation
#         mask or not.
#       camera_id: The camera id to be passed to OpenCV.
#       width: The width of the frame captured from the camera.
#       height: The height of the frame captured from the camera.
#   """

#     # Start capturing video input from the camera
#     cap = cv2.VideoCapture(camera_id)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#     # Visualization parameters
#     row_size = 50  # pixels
#     left_margin = 24  # pixels
#     text_color = (0, 0, 0)  # black
#     font_size = 1
#     font_thickness = 1
#     fps_avg_frame_count = 10
#     overlay_alpha = 0.5
#     mask_color = (100, 100, 0)  # cyan

#     def save_result(result: vision.PoseLandmarkerResult,
#                     unused_output_image: mp.Image, timestamp_ms: int):
#         global FPS, COUNTER, START_TIME, pose_results

#         # Calculate the FPS
#         if COUNTER % fps_avg_frame_count == 0:
#             FPS = fps_avg_frame_count / (time.time() - START_TIME)
#             START_TIME = time.time()

#         pose_results = result
#         COUNTER += 1

#     # Initialize the pose landmarker model
#     base_options = python.BaseOptions(model_asset_path=model)
#     options = vision.PoseLandmarkerOptions(
#         base_options=base_options,
#         running_mode=vision.RunningMode.LIVE_STREAM,
#         num_poses=num_poses,
#         min_pose_detection_confidence=min_pose_detection_confidence,
#         min_pose_presence_confidence=min_pose_presence_confidence,
#         min_tracking_confidence=min_tracking_confidence,
#         output_segmentation_masks=output_segmentation_masks,
#         result_callback=save_result)
#     pose_detector = vision.PoseLandmarker.create_from_options(options)

#     # Continuously capture images from the camera and run inference
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             sys.exit(
#                 'ERROR: Unable to read from webcam. Please verify your webcam settings.'
#             )

#         image = cv2.flip(image, 1)

#         # Convert the image from BGR to RGB as required by the TFLite model.
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

#         # Run pose landmarker using the model.
#         pose_detector.detect_async(mp_image, time.time_ns() // 1_000_000)

#         # Show the FPS
#         fps_text = 'FPS = {:.1f}'.format(FPS)
#         text_location = (left_margin, row_size)
#         current_frame = image
#         cv2.putText(current_frame, fps_text, text_location,
#                     cv2.FONT_HERSHEY_DUPLEX,
#                     font_size, text_color, font_thickness, cv2.LINE_AA)

#         if pose_results:
#             # Draw landmarks.
#             for pose_landmarks in pose_results.pose_landmarks:
#                 # Draw the pose landmarks.
#                 pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#                 pose_landmarks_proto.landmark.extend([
#                     landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
#                                                     z=landmark.z) for landmark
#                     in pose_landmarks
#                 ])
#                 mp_drawing.draw_landmarks(
#                     current_frame,
#                     pose_landmarks_proto,
#                     mp_pose.POSE_CONNECTIONS,
#                     mp_drawing_styles.get_default_pose_landmarks_style())

#         if (output_segmentation_masks and pose_results):
#             if pose_results.segmentation_masks is not None:
#                 segmentation_mask = pose_results.segmentation_masks[0].numpy_view()
#                 mask_image = np.zeros(image.shape, dtype=np.uint8)
#                 mask_image[:] = mask_color
#                 condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
#                 visualized_mask = np.where(condition, mask_image, current_frame)
#                 current_frame = cv2.addWeighted(current_frame, overlay_alpha,
#                                                 visualized_mask, overlay_alpha,
#                                                 0)

#         cv2.imshow('pose_landmarker', current_frame)

#         # Stop the program if the ESC key is pressed.
#         if cv2.waitKey(1) == 27:
#             break

#     pose_detector.close()
#     cap.release()
#     cv2.destroyAllWindows()


# def main():
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         '--model',
#         help='Name of the pose landmarker model bundle.',
#         required=False,
#         default='pose_landmarker.task')
#     parser.add_argument(
#         '--numPoses',
#         help='Max number of poses that can be detected by the landmarker.',
#         required=False,
#         default=1)
#     parser.add_argument(
#         '--minPoseDetectionConfidence',
#         help='The minimum confidence score for pose detection to be considered '
#              'successful.',
#         required=False,
#         default=0.5)
#     parser.add_argument(
#         '--minPosePresenceConfidence',
#         help='The minimum confidence score of pose presence score in the pose '
#              'landmark detection.',
#         required=False,
#         default=0.5)
#     parser.add_argument(
#         '--minTrackingConfidence',
#         help='The minimum confidence score for the pose tracking to be '
#              'considered successful.',
#         required=False,
#         default=0.5)
#     parser.add_argument(
#         '--outputSegmentationMasks',
#         help='Set this if you would also like to visualize the segmentation '
#              'mask.',
#         required=False,
#         action='store_true')
#     # Finding the camera ID can be very reliant on platform-dependent methods.
#     # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
#     # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
#     # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
#     parser.add_argument(
#         '--cameraId', help='Id of camera.', required=False, default=0)
#     parser.add_argument(
#         '--frameWidth',
#         help='Width of frame to capture from camera.',
#         required=False,
#         default=1280)
#     parser.add_argument(
#         '--frameHeight',
#         help='Height of frame to capture from camera.',
#         required=False,
#         default=960)
#     args = parser.parse_args()

#     run(args.model, int(args.numPoses), args.minPoseDetectionConfidence,
#         args.minPosePresenceConfidence, args.minTrackingConfidence,
#         args.outputSegmentationMasks,
#         int(args.cameraId), args.frameWidth, args.frameHeight)
