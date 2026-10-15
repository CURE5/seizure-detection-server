# Seizure Detection Architecture Overview

## System Components

### 1. Model Layer (ONNX)
**Purpose:** Detect individual spasms from pose sequences

**Input:** 30-frame sequences (6 seconds at 5 FPS) with 33 MediaPipe landmarks per frame
**Output:** Confidence score (0-1) indicating likelihood of hypermotor spasm
**Performance:** 0.79ms inference, 99.6% recall, 98.1% precision (validated on 1998 test sequences)
**ROC AUC:** 0.9968 (excellent discrimination)

**What it does:**
- Analyzes pose sequences to detect abnormal movement patterns
- Trained on 121 seizure videos, 56 normal activity videos
- Detects individual spasms, **not** seizure patterns

**What it doesn't do:**
- Understand time relationships between spasms
- Recognize rhythmic patterns
- Track seizure duration or state

---

### 2. Server Layer (Python/FastAPI)
**Purpose:** Interpret model outputs over time

**Components:**

#### A. Multi-Stage Pipeline (`pipeline.py`)
- Stage 1: Motion detection (skip frames with no movement)
- Stage 2: Activity analysis (filter normal activities)
- Stage 3: ONNX model inference (detect spasms)
- Stage 4: Rhythmic pattern recognition (interpret as seizure)

#### B. Pattern Recognition (Server-Side Post-Processing)
**What it does:**
- Tracks timestamps of each spasm detection
- Calculates intervals between spasms
- Recognizes calm periods (4-20s) as part of seizure rhythm
- Adapts thresholds based on observed patterns
- Determines seizure onset/offset

**Two-Level Timeout System:**
1. **Intra-seizure gaps** (default 30s, adaptive):
   - Allows calm periods between spasms
   - Adapts to observed rhythm (2x average interval, max 45s)
   - Prevents false seizure-end during normal pauses

2. **Seizure end threshold** (60s):
   - Requires 60+ seconds of no spasms
   - Declares true seizure offset

**Example Scenario:**
```
Model says:          Server interprets:
10:00:05 → Spasm!    → Seizure ONSET (first spasm)
10:00:12 → Spasm!    → Seizure ONGOING (7s gap - normal)
10:00:28 → Spasm!    → Seizure ONGOING (16s gap - rhythmic)
10:00:45 → Spasm!    → Seizure ONGOING (17s gap - adaptive threshold)
10:02:00 → ...       → Seizure OFFSET (60s+ silence)

Result: Single 1m55s seizure with 4 spasms, avg 13.3s interval
```

---

## Why This Architecture?

### Separation of Concerns

**Model (ONNX):**
- Fast, efficient spasm detection
- No temporal state or context
- Static inference (same input → same output)

**Server (Python):**
- Temporal pattern recognition
- Stateful tracking (remembers history)
- Configurable thresholds (no retraining needed)

### Benefits

1. **No Retraining Required:**
   - Pattern logic tunable via `.env` variables
   - Can adjust thresholds without touching model
   - Server code updates don't require model re-export

2. **Domain Knowledge Integration:**
   - Hypermotor seizure characteristics baked into server logic
   - Understands typical 4-20s calm periods
   - Adapts to individual seizure patterns

3. **Maintainability:**
   - Model stays simple and fast
   - Complex pattern logic in readable Python code
   - Easy to debug and adjust thresholds

4. **Flexibility:**
   - Can add new pattern types without retraining
   - Support different seizure types with different thresholds
   - A/B test different strategies in production

---

## Configuration

### Model Settings (Inference)
```bash
# .env
USE_POSE_MODEL=true
POSE_MODEL_PATH=models/hypermotor_detector.onnx
POSE_MODEL_THRESHOLD=0.5          # Min confidence for spasm detection
                                   # Validated: 99.6% recall, 98.1% precision
                                   # ROC AUC: 0.9968 on test set
```

**Threshold Validation:**
The 0.5 threshold has been validated through comprehensive ROC/PR curve analysis on 1998 test sequences:
- Only 6 missed seizures out of 1519 (99.6% recall)
- Only 29 false alarms (98.1% precision)
- Near-optimal operating point (max F1 score at 0.874)
- For maximum safety, can lower to 0.20-0.30 (100% recall, more false alarms)
- Re-analyze with: `python scripts/analyze_thresholds.py`

### Pattern Recognition (Server)
```bash
# .env
INTRA_SEIZURE_MAX_GAP=30.0        # Base threshold for calm periods
SEIZURE_END_THRESHOLD=60.0         # Min silence for seizure end
ALERT_STRATEGY=onset_only          # When to send alerts
ALERT_COOLDOWN_SECONDS=30          # Min time between alerts
```

**No model changes needed** - all pattern tuning happens server-side.

---

## Data Flow

```
Camera Frame
    ↓
Pose Detection (MediaPipe/Hailo)
    ↓ (33 landmarks)
Multi-Stage Pipeline
    ↓ (Stage 1: Motion)
    ↓ (Stage 2: Activity)
    ↓ (Stage 3: ONNX Model)
    ↓ (confidence score)
Pattern Recognition
    ↓ (track timestamps, intervals)
    ↓ (apply two-level timeouts)
    ↓ (determine onset/offset)
Alert Decision
    ↓
Notifications + Recording + Logging
```

**Key Point:** Model runs once per frame, pattern recognition runs continuously over time.

---

## When to Retrain Model

You **DO need** to retrain if:
- ❌ Model accuracy drops (too many missed spasms)
- ❌ Different seizure types need detection
- ❌ New camera angle or environment
- ❌ The subject's movement patterns change significantly

You **DON'T need** to retrain if:
- ✅ Adjusting timing thresholds (intra-seizure gaps, seizure end)
- ✅ Changing alert strategy (onset only, offset, both)
- ✅ Tuning pattern recognition logic
- ✅ Adding new notification backends
- ✅ Modifying recording behavior

---

## Summary

**Model = "What" (individual spasms)**
**Server = "When & How Long" (seizure patterns)**

This architecture allows sophisticated seizure pattern recognition without model complexity, making the system both accurate and maintainable.
