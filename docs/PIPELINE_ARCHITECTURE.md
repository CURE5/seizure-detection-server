# Multi-Stage Detection Pipeline

## Overview

The multi-stage detection pipeline implements a cascading approach to seizure detection that balances **accuracy**, **performance**, and **false positive reduction**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Video Frame (5 FPS)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────┐
    │  STAGE 1: Motion Detection                       │
    │  - Fast frame differencing                       │
    │  - Motion degree threshold: 2.5%                 │
    │  - Purpose: Skip analysis if no movement         │
    └────────────┬─────────────────────────────────────┘
                 │ Motion detected?
                 │ No → Skip (save computation)
                 │ Yes ↓
    ┌──────────────────────────────────────────────────┐
    │  STAGE 2: Activity Analysis                      │
    │  - Generic pose-based detection                  │
    │  - Velocity/acceleration analysis                │
    │  - Rhythmic motion detection                     │
    │  - Confidence threshold: 0.7                     │
    │  - Purpose: Filter normal activities             │
    └────────────┬─────────────────────────────────────┘
                 │ Suspicious activity?
                 │ No → Skip (reduce false positives)
                 │ Yes ↓
    ┌──────────────────────────────────────────────────┐
    │  STAGE 3: Custom ONNX Model                      │
    │  - Patient-specific trained model                │
    │  - 30-frame sequence (6 seconds)                 │
    │  - Hypermotor seizure detection                  │
    │  - 100% recall, 76% precision                    │
    │  - Purpose: Final classification                 │
    └────────────┬─────────────────────────────────────┘
                 │ Seizure detected?
                 │ No → Continue monitoring
                 │ Yes ↓
    ┌──────────────────────────────────────────────────┐
    │  Confirmation Logic                              │
    │  - Requires 2/5 consecutive detections           │
    │  - Time-of-day weighting (5-8 AM = 1.5x)        │
    │  - Alert cooldown: 30 seconds                    │
    └────────────┬─────────────────────────────────────┘
                 │ Confirmed?
                 │ Yes ↓
    ┌──────────────────────────────────────────────────┐
    │  🚨 SEIZURE ALERT                                │
    │  - Notify caregivers                             │
    │  - Start video recording                         │
    │  - Log event                                     │
    └──────────────────────────────────────────────────┘
```

## Stage Details

### Stage 1: Motion Detection

**Purpose:** Fast screening to skip analysis when no movement occurs.

**Method:**
- Frame differencing (grayscale)
- Motion degree = changed pixels / total pixels
- Threshold: 2.5% (adjustable)

**Benefits:**
- Very fast (<1ms per frame)
- Saves computation during sleep
- No false positives (only filters out)

**Typical Results:**
- 70-90% of frames pass (during active monitoring)
- 10-30% of frames pass (during sleep)

### Stage 2: Activity Analysis

**Purpose:** Filter normal activities and detect generic seizure indicators.

**Method:**
- Track 8 key landmarks (shoulders, elbows, wrists, hips)
- Calculate velocity and acceleration
- Detect rhythmic motion (2-5 Hz)
- Detect sudden movements
- Combined confidence score

**Thresholds:**
- Velocity: 0.15 m/s
- Acceleration: 0.25 m/s²
- Confidence: 0.7 (with time weighting)

**Benefits:**
- Works without custom model
- Catches non-hypermotor seizures
- Reduces false positives from normal activities

**Typical Results:**
- 20-40% of motion frames pass to Stage 3
- Filters out: walking, stretching, adjusting position

### Stage 3: Custom ONNX Model

**Purpose:** Patient-specific hypermotor seizure detection.

**Method:**
- Bidirectional LSTM with attention
- 30-frame sequences (6 seconds at 5 FPS)
- 17 COCO keypoints per frame
- Trained on 121 seizure + 56 normal videos

**Performance:**
- **Recall: 100%** (zero missed seizures)
- **Precision: 76%** (acceptable false positive rate)
- **F1 Score: 86.4%**
- **Inference: 0.79ms** (1,267 FPS throughput)

**Benefits:**
- Patient-specific patterns
- Very fast inference
- High sensitivity (safety-critical)

## Time-of-Day Weighting

Early morning (5-8 AM) is high-risk period for hypermotor seizures.

**Sensitivity Multipliers:**
- **5:00 AM - 8:00 AM:** 1.5x (high risk)
- **4:00 AM - 5:00 AM:** 1.25x (moderate)
- **8:00 AM - 9:00 AM:** 1.25x (moderate)
- **Rest of day:** 1.0x (normal)

**Effect:**
- Lowers detection threshold during high-risk hours
- Increases sensitivity when most needed
- Reduces false negatives during critical period

## Confirmation Logic

To reduce false alarms, the system requires **confirmation**:

**Rules:**
- Minimum 2 positive detections in last 5 frames
- Effectively: 2-3 consecutive detections
- Alert cooldown: 30 seconds between alerts

**Example:**
```
Frame sequence: [-, -, +, +, -]  → CONFIRMED (2/5)
Frame sequence: [-, +, -, -, -]  → NOT CONFIRMED (1/5)
Frame sequence: [+, +, +, -, -]  → CONFIRMED (3/5)
```

## Rhythmic Seizure Pattern Detection

### Server-Side Post-Processing (No Model Changes)

The system includes intelligent **rhythmic pattern recognition** for hypermotor seizures:

**Separation of Concerns:**
- **Model (ONNX):** Detects individual spasms from 30-frame pose sequences
- **Server:** Interprets multiple spasms over time as rhythmic seizure pattern

This architecture means:
- ✅ No model retraining required for pattern detection
- ✅ Thresholds tunable via environment variables
- ✅ Pattern logic can be updated without touching model

**Two-Level Timeout System:**

1. **Intra-seizure gaps** (default 30s, adaptive up to 45s):
   - Allows 4-20s calm periods between spasms
   - Adapts based on observed rhythm (2x average interval)
   - Prevents false seizure-end during normal rhythmic pauses

2. **Seizure end threshold** (default 60s):
   - Requires 60+ seconds of no activity to declare seizure ended
   - Ensures complete seizure offset, not just temporary calm

**Pattern Tracking:**
- Records timestamp of each spasm detection
- Calculates intervals between spasms
- Adapts max gap based on observed pattern
- Logs: spasm count, average interval, rhythmic patterns

**Configuration** (`.env`):
```bash
INTRA_SEIZURE_MAX_GAP=30.0    # Base threshold for calm periods
SEIZURE_END_THRESHOLD=60.0     # Minimum silence for seizure end
POSE_MODEL_THRESHOLD=0.5       # Spasm detection confidence
```

**Model Performance (Validated on 1998 test sequences):**
- ROC AUC: 0.9968
- Recall: 99.6% (only 6 missed seizures out of 1519)
- Precision: 98.1% (29 false alarms out of 1998 predictions)
- Threshold 0.5 is near-optimal, validated through ROC/PR curve analysis

**Example:**
```
10:00:05 - Spasm detected (onset)
10:00:12 - Spasm detected (7s gap - normal rhythm)
10:00:28 - Spasm detected (16s gap - still within threshold)
10:00:45 - Spasm detected (17s gap - adaptive threshold allows)
10:02:00 - 60s+ elapsed → Seizure ended (offset)
```

Result: Single 1m55s seizure with 4 spasms (avg 13.3s interval), not 4 separate seizures.

## Performance Analysis

### Computational Efficiency

| Stage | Cost/Frame | Pass Rate | Cumulative Cost |
|-------|------------|-----------|-----------------|
| Motion | ~1ms | 70% | 1ms |
| Activity | ~5ms | 30% | 2.5ms |
| Custom | ~1ms | 15% | 1.15ms |
| **Total** | - | - | **~4.65ms** |

**At 5 FPS:** ~23ms per second (well below 200ms budget)

### False Positive Reduction

Without pipeline (custom model alone):
- False positive rate: ~24% (from 76% precision)
- ~1 false alarm per 4 detections

With pipeline (3-stage + confirmation):
- Estimated false positive rate: ~5-10%
- ~1 false alarm per 10-20 detections
- **70-85% reduction in false positives**

## Usage

### Basic Usage

```python
from pathlib import Path
from seizure_detection.analysis import MultiStageDetector
from seizure_detection.config import SeizureDetectionConfig

# Initialize pipeline
config = SeizureDetectionConfig()
detector = MultiStageDetector(
    config=config,
    onnx_model_path=Path('models/hypermotor_detector.onnx'),
    enable_time_weighting=True
)

# Process frames
while True:
    frame = get_video_frame()  # BGR format
    landmarks = get_pose_landmarks()  # From Hailo or MediaPipe
    timestamp = datetime.now()
    
    # Analyze
    result = detector.analyze_frame(frame, landmarks, timestamp)
    
    # Check for alerts
    if result['alert']:
        send_notification()
        start_recording()
```

### Result Structure

```python
result = {
    'timestamp': '2025-01-15T05:30:00',
    'time_weight': 1.5,  # 5:30 AM = high risk
    'stage_reached': 3,  # Made it to custom model
    
    'motion': {
        'detected': True,
        'degree': 0.035,
        'threshold': 0.025
    },
    
    'activity': {
        'detected': True,
        'confidence': 0.82,
        'indicators': {
            'rhythmic_motion': {'detected': True, 'confidence': 0.75},
            'sudden_movement': {'detected': True, 'confidence': 0.65},
            'high_acceleration': {'detected': False, 'confidence': 0.0}
        }
    },
    
    'custom': {
        'ready': True,
        'detected': True,
        'probability': 0.89,
        'threshold': 0.5,
        'consecutive_detections': 3
    },
    
    'confirmed': True,  # Met confirmation criteria
    'alert': True  # Send alert!
}
```

### Statistics

```python
stats = detector.get_statistics()
# {
#     'motion_triggers': 12500,
#     'activity_triggers': 3200,
#     'custom_triggers': 480,
#     'confirmed_seizures': 3,
#     'efficiency': {
#         'motion_pass_rate': 0.256,  # 25.6% pass Stage 1
#         'activity_pass_rate': 0.15   # 15% pass Stage 2
#     }
# }
```

## Tuning Parameters

### Sensitivity vs False Positives

**More Sensitive (fewer missed seizures, more false alarms):**
- Lower `confidence_threshold` in custom detector (0.4 instead of 0.5)
- Lower `CONFIDENCE_THRESHOLD` in config (0.6 instead of 0.7)
- Increase time-of-day weight (1.75x instead of 1.5x)
- Require 1/5 confirmations instead of 2/5

**Fewer False Positives (more selective):**
- Raise `confidence_threshold` to 0.6-0.7
- Raise `CONFIDENCE_THRESHOLD` to 0.8
- Require 3/5 confirmations
- Increase alert cooldown to 60 seconds

### Current Settings (Recommended)

These settings achieve **100% recall** with **acceptable false positives**:

```python
# Stage 3: Custom model
confidence_threshold = 0.5

# Stage 2: Activity analysis
CONFIDENCE_THRESHOLD = 0.7

# Time weighting
high_risk_multiplier = 1.5  # 5-8 AM

# Confirmation
min_detections = 2  # out of 5 frames
alert_cooldown = 30  # seconds
```

## Deployment Strategy

### Phase 1: Shadow Mode (Week 1-2)

- Pipeline runs but **doesn't send alerts**
- Log all detections for analysis
- Collect 24/7 monitoring data
- Identify false positive patterns
- Tune thresholds based on results

### Phase 2: Testing Mode (Week 3-4)

- Enable alerts during **high-risk hours only** (5-8 AM)
- Parent validation: confirm real vs false positives
- Adjust sensitivity based on feedback
- Monitor caregiver burden (false alarms)

### Phase 3: Production (Week 5+)

- Enable 24/7 alerts
- Continue logging for model improvement
- Periodic review of false positive rate
- Collect new seizure videos for retraining

## Future Enhancements

### Short-term
1. **Adaptive thresholds** based on historical data
2. **Motion-specific filters** (e.g., ignore if parent enters room)
3. **Multi-camera fusion** (if second camera available)

### Medium-term
1. **Hailo AI integration** (hardware-accelerated Stage 3)
2. **Online learning** (update model with new seizures)
3. **Confidence visualization** (show detection strength to caregivers)

### Long-term
1. **Seizure forecasting** (predict high-risk periods)
2. **Multi-patient models** (improve generalization)
3. **Integration with medical devices** (e.g., EEG, EMG)

## Testing

Test the pipeline:

```bash
# Test architecture
python scripts/test_pipeline.py --architecture-only

# Test with video
python scripts/test_pipeline.py \
    --video training_data/seizure/videos/video_001.mp4 \
    --model models/hypermotor_detector.onnx
```

## References

- **Training Results:** See `models/` directory
- **Model Export:** `scripts/export_to_onnx.py`
