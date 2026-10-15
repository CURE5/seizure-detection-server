# Hailo AI Integration for Seizure Detection

## Overview

This document explains how Hailo AI hardware acceleration is used for real-time seizure detection on Raspberry Pi 5.

## Architecture

```
┌─────────────────┐
│  Video Frame    │
│  (640x480 BGR)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Hailo-8L AI Accelerator        │
│  YOLOv8-Pose Model              │
│  (yolov8s_pose.hef)            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  17 COCO Keypoints per Person   │
│  Format: [x, y, confidence]×17  │
│                                 │
│  0: Nose         9: L Wrist     │
│  1: L Eye       10: R Wrist     │
│  2: R Eye       11: L Hip       │
│  3: L Ear       12: R Hip       │
│  4: R Ear       13: L Knee      │
│  5: L Shoulder  14: R Knee      │
│  6: R Shoulder  15: L Ankle     │
│  7: L Elbow     16: R Ankle     │
│  8: R Elbow                     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Normalization & Conversion     │
│  Pixel → 0-1 Range              │
│  COCO → Landmark Format         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  SeizureDetector                │
│  (seizure_detector.py)          │
│                                 │
│  Analyzes:                      │
│  • Rhythmic Motion (2-5 Hz)     │
│  • Velocity Changes             │
│  • Acceleration Spikes          │
│  • Posture Shifts               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Seizure Alert                  │
│  Confidence Score: 0.0-1.0      │
└─────────────────────────────────┘
```

## Hailo AI Advantages for RPi 5

### Performance Benefits

| Metric | MediaPipe (CPU) | Hailo AI (NPU) | Improvement |
|--------|----------------|----------------|-------------|
| FPS | 5-10 FPS | 30-60 FPS | **6-12x faster** |
| CPU Usage | 60-80% | 15-25% | **3-4x lower** |
| Latency | 100-200ms | 16-33ms | **6x lower** |
| Power | ~5W extra | ~2W extra | **2.5x efficient** |

### Why This Matters for Seizure Detection

1. **Lower Latency** = Faster alerts during critical moments
2. **Higher FPS** = More data points for rhythm detection (2-5 Hz requires 10+ FPS)
3. **Lower CPU** = More resources for video recording, logging, alerting
4. **Power Efficient** = Can run 24/7 without overheating

## Seizure Detection Algorithm

### 1. Rhythmic Motion Detection (Primary Indicator)

**How it works:**
```python
# Track wrist positions over time (30 frames @ 5 FPS = 6 seconds)
wrist_positions = deque(maxlen=30)

for frame in video_stream:
    left_wrist = pose_data['landmarks'][9]  # Left wrist
    right_wrist = pose_data['landmarks'][10]  # Right wrist
    
    wrist_positions.append({
        'left': (left_wrist['x'], left_wrist['y']),
        'right': (right_wrist['x'], right_wrist['y']),
        'timestamp': time.time()
    })
    
    # Calculate oscillation frequency using FFT
    frequencies = fft(wrist_positions)
    
    # Check for rhythmic pattern in seizure range (2-5 Hz)
    if peak_frequency in range(2, 5):
        rhythmic_score = 0.8  # High confidence
```

**Why wrists?**
- Most visible during tonic-clonic seizures
- Large amplitude movements
- Bilateral symmetry (both sides shake)

### 2. Velocity Analysis (Secondary Indicator)

**How it works:**
```python
# Calculate frame-to-frame velocity
previous_position = None

for frame in video_stream:
    current_position = get_key_landmarks(pose_data)
    
    if previous_position:
        # Euclidean distance between frames
        velocity = distance(current_position, previous_position) / time_delta
        
        # Sudden velocity spikes indicate loss of motor control
        if velocity > VELOCITY_THRESHOLD:
            velocity_score = 0.6
    
    previous_position = current_position
```

**Thresholds:**
- Normal movement: 0.01-0.05 (normalized units/second)
- Seizure movement: 0.15+ (3-15x faster)

### 3. Acceleration Detection (Tertiary Indicator)

**How it works:**
```python
# Second derivative of position = acceleration
velocities = deque(maxlen=3)

for frame in video_stream:
    velocity = calculate_velocity(current, previous)
    velocities.append(velocity)
    
    if len(velocities) >= 2:
        acceleration = (velocities[-1] - velocities[-2]) / time_delta
        
        # Sharp jerks have high acceleration
        if abs(acceleration) > ACCELERATION_THRESHOLD:
            acceleration_score = 0.7
```

**Physical interpretation:**
- High acceleration = sudden start/stop of movement
- Characteristic of myoclonic jerks
- Multiple spikes = clonic phase of seizure

### 4. Posture Change Detection (Safety Indicator)

**How it works:**
```python
# Monitor body alignment
shoulders = (landmarks[5], landmarks[6])  # L/R shoulders
hips = (landmarks[11], landmarks[12])      # L/R hips

# Calculate body angle from vertical
body_angle = calculate_angle(shoulders, hips, vertical_axis)

# Detect falling or collapse
if abs(body_angle) > 45:  # More than 45° from vertical
    posture_score = 0.9  # Very concerning
```

**Why this matters:**
- Falling is immediate safety risk
- Loss of postural control = severe seizure
- Triggers high-priority alert

## Confidence Scoring

The system combines all indicators into a final confidence score:

```python
confidence = weighted_average([
    (rhythmic_score, 0.40),      # 40% weight - primary indicator
    (velocity_score, 0.25),      # 25% weight - secondary
    (acceleration_score, 0.20),  # 20% weight - tertiary  
    (posture_score, 0.15)        # 15% weight - safety
])

if confidence > 0.7:
    trigger_alert("SEIZURE DETECTED", confidence)
```

## Code Flow in `mediapipe_analysis.py`

### 1. Initialization (`setup()` function)

```python
# Lines 121-157
pose_detector = PoseEstimation(
    hef_path="yolov8s_pose.hef",
    threshold=0.5,               # Min confidence for detection
    nms_iou_threshold=0.7        # Suppress overlapping detections
)
```

**What this does:**
- Loads YOLOv8-Pose model compiled for Hailo-8L
- Configures detection thresholds
- Initializes hardware accelerator connection

### 2. Frame Processing (`process_frame()` function)

```python
# Lines 390-502
detections = pose_detector.process(image)  # Hardware inference

# Extract keypoints
keypoints = detection.get('keypoints', [])  # [x,y,conf] × 17

# Normalize coordinates
norm_x = x / image.shape[1]  # Pixel → 0-1 range
norm_y = y / image.shape[0]

# Package for SeizureDetector
pose_data = {'pose_landmarks': [pose_landmarks]}
```

**What this does:**
- Runs frame through Hailo hardware
- Gets 17 COCO keypoints with confidence scores
- Normalizes to 0-1 range for consistency
- Returns standardized format

### 3. Data Pipeline to Seizure Detector

```python
# In server.py (main processing loop)
frame, data = mediapipe_analysis.process_frame(frame)

if data['pose']:
    # Feed to seizure detector
    seizure_detector.add_pose_data(
        landmarks=data['pose']['pose_landmarks'],
        timestamp=time.time()
    )
    
    # Analyze for seizures
    result = seizure_detector.analyze()
    
    if result['detected']:
        # Trigger alert
        logger.critical(f"SEIZURE DETECTED: {result['confidence']:.2f}")
        start_recording()
        send_notification()
```

## Key Differences: Hailo vs MediaPipe

| Aspect | MediaPipe | Hailo AI |
|--------|-----------|----------|
| **Hardware** | CPU | Hailo-8L NPU |
| **Model** | BlazePose | YOLOv8-Pose |
| **Keypoints** | 33 (full body) | 17 (COCO) |
| **Depth (Z)** | Yes | No (2D only) |
| **Gaze** | Yes (face mesh) | No |
| **FPS (RPi5)** | 5-10 | 30-60 |
| **Accuracy** | Higher (33 points) | Good (17 points) |
| **Use Case** | Development/Testing | Production on RPi5 |

**For Seizure Detection:**
- 17 COCO keypoints are sufficient (we primarily use wrists, shoulders, hips, ankles)
- Lack of Z-depth is acceptable (2D motion patterns are enough)
- No gaze tracking is fine (not critical for seizure detection)
- **Speed is critical** - 30+ FPS allows accurate rhythm detection

## Configuration

### Model Files

```bash
# Hailo model location (must be downloaded separately)
/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef

# Download from:
# https://github.com/hailo-ai/hailo_model_zoo
```

### Tuning Parameters

```python
# In mediapipe_analysis.py
pose_detector = PoseEstimation(
    hef_path="yolov8s_pose.hef",
    
    # Detection confidence (0.0-1.0)
    # Lower = more sensitive but more false positives
    # Higher = more accurate but may miss detections
    threshold=0.5,  # Default: 0.5
    
    # Non-max suppression (0.0-1.0)
    # Controls overlapping detection removal
    # Lower = keep more detections
    # Higher = more aggressive filtering
    nms_iou_threshold=0.7  # Default: 0.7
)
```

### Seizure Detection Thresholds

```python
# In seizure_detection/config.py
SEIZURE_DETECTION_CONFIG = {
    'rhythmic_frequency_range': (2, 5),  # Hz
    'velocity_threshold': 0.15,          # Normalized units/sec
    'acceleration_threshold': 0.5,       # Normalized units/sec²
    'posture_angle_threshold': 45,       # Degrees from vertical
    'confidence_threshold': 0.7,         # 0.0-1.0
    'history_length': 30                 # Frames (~6 sec @ 5 FPS)
}
```

## Testing Recommendations

### 1. Hardware Requirements
- Raspberry Pi 5 (4GB+ RAM recommended)
- Hailo-8L AI Kit installed
- Camera module or USB camera

### 2. Performance Testing

```bash
# Monitor FPS and CPU usage
python server.py --log-level DEBUG

# Expected output:
# INFO: Hailo AI backend initialized
# INFO: Processing at 45 FPS, CPU: 22%
```

### 3. Accuracy Testing

Use test videos with known seizure timestamps:

```python
# Test with known seizure video
VIDEO_PATH = "test_videos/seizure_2024_01_15.mp4"

# Compare detection timestamps with ground truth
# Calculate: True Positives, False Positives, False Negatives
```

### 4. Benchmark Comparison

```bash
# Test MediaPipe backend
BACKEND=mediapipe python server.py

# Test Hailo backend  
BACKEND=hailo python server.py

# Compare:
# - FPS, CPU usage, accuracy, latency
```

## Future Improvements

### Short-term
1. **Optimize threshold values** based on real-world testing
2. **Add temporal smoothing** to reduce false positives
3. **Implement multi-person tracking** (currently uses first detected person)

### Long-term
1. **Train custom Hailo model** fine-tuned on seizure data
2. **Add audio analysis** (vocalization during seizures)
3. **Implement ML classifier** trained on historical seizure patterns
4. **Edge AI inference** for privacy-preserving cloud backup

## References

- [Hailo-8L AI Accelerator Datasheet](https://hailo.ai/products/hailo-8l-ai-accelerator/)
- [YOLOv8-Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)
- [Hailo RPi5 Examples](https://github.com/hailo-ai/hailo-rpi5-examples)


## Local training example

    https://github.com/DeGirum/hailo_examples/blob/main/user_guides/005_docker.md

---

**Document Version:** 1.0  
**Date:** December 2, 2025  
