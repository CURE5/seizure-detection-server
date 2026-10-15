# MediaPipe 33 Pose Landmarks

## Overview

The hypermotor seizure detection model now uses **all 33 MediaPipe Pose landmarks** instead of the reduced 17 COCO keypoints. This preserves full movement fidelity for accurate seizure detection.

## Why 33 Landmarks?

Hypermotor seizures involve complex whole-body movements:
- **Hand thrashing** - requires finger/wrist detail
- **Foot kicking** - requires heel/toe/ankle detail  
- **Facial movements** - mouth, eye coordination
- **Asymmetric patterns** - more points = better asymmetry detection

The previous 17-keypoint COCO format lost critical detail by discarding:
- Hand landmarks (index, pinky, thumb)
- Foot landmarks (heel, foot_index)
- Mouth corners
- Eye detail variations

## MediaPipe 33 Landmark Structure

```
Index | Landmark Name      | Body Part
------|-------------------|------------------
0     | nose              | Face
1     | left_eye_inner    | Face
2     | left_eye          | Face  
3     | left_eye_outer    | Face
4     | right_eye_inner   | Face
5     | right_eye         | Face
6     | right_eye_outer   | Face
7     | left_ear          | Face
8     | right_ear         | Face
9     | mouth_left        | Face
10    | mouth_right       | Face
11    | left_shoulder     | Upper Body
12    | right_shoulder    | Upper Body
13    | left_elbow        | Upper Body
14    | right_elbow       | Upper Body
15    | left_wrist        | Upper Body
16    | right_wrist       | Upper Body
17    | left_pinky        | Hands
18    | right_pinky       | Hands
19    | left_index        | Hands
20    | right_index       | Hands
21    | left_thumb        | Hands
22    | right_thumb       | Hands
23    | left_hip          | Lower Body
24    | right_hip         | Lower Body
25    | left_knee         | Lower Body
26    | right_knee        | Lower Body
27    | left_ankle        | Lower Body
28    | right_ankle       | Lower Body
29    | left_heel         | Feet
30    | right_heel        | Feet
31    | left_foot_index   | Feet
32    | right_foot_index  | Feet
```

## Data Format

### Extracted Sequences
- **Shape**: `(30, 33, 3)`
  - 30 frames (6 seconds at 5 FPS)
  - 33 keypoints
  - 3 features per keypoint: `(x, y, visibility)`

### Feature Values
- **x, y**: Normalized coordinates [0.0 - 1.0]
- **visibility**: Confidence score [0.0 - 1.0]

## Model Architecture Changes

### Input
```python
Input: (batch, 30, 33, 3)
```

### After Flattening
```python
Flattened: (batch, 30, 99)
# 33 keypoints × 3 features = 99 features per frame
```

### Model Parameters
- **LSTM input_size**: 99 (was 51)
- **Hidden size**: 128 (unchanged)
- **Output**: 1 (seizure probability)

## Training Pipeline

### 1. Extract Poses
```bash
python scripts/extract_poses_direct.py \
    --seizure-dir training_data/seizure/videos \
    --normal-dir training_data/normal/videos \
    --output-dir training_data/
```

Output: `.npy` files with shape `(30, 33, 3)`

### 2. Train Model
```bash
python scripts/train_seizure_model.py \
    --data-dir training_data/ \
    --splits-file training_data/dataset_splits.json \
    --output-dir models/ \
    --epochs 50 \
    --batch-size 32 \
    --hidden-size 128
```

### 3. Export to ONNX
```bash
python scripts/export_to_onnx.py \
    --model-path models/hypermotor_detector_best.pth \
    --output-path models/hypermotor_detector.onnx
```

## Deployment Note

⚠️ **This model REQUIRES MediaPipe for inference** - it cannot use Hailo YOLOv8-Pose because:
- YOLOv8-Pose outputs only 17 COCO keypoints
- MediaPipe provides 33 landmarks
- The model is trained on 33-point sequences

To use this model in production:
```python
from seizure_detection.config import FORCE_MEDIAPIPE

# Force MediaPipe backend
FORCE_MEDIAPIPE = True
```

## Benefits

1. **Higher accuracy** - Captures subtle movements in hands, feet, face
2. **Better asymmetry detection** - More points = better left/right comparison
3. **Richer temporal patterns** - LSTM can learn more complex dynamics
4. **Patient-specific tuning** - More features = better personalization

## Comparison: 17 vs 33 Keypoints

| Feature | 17 COCO | 33 MediaPipe | Improvement |
|---------|---------|--------------|-------------|
| Face detail | 5 points | 11 points | +120% |
| Hand detail | 2 points (wrists) | 8 points (wrists+fingers) | +300% |
| Foot detail | 2 points (ankles) | 6 points (ankles+heels+toes) | +200% |
| Total features/frame | 51 | 99 | +94% |
| Movement fidelity | Reduced | Full | ✓ |

## References

- [MediaPipe Pose Landmark Model](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe BlazePose Paper](https://arxiv.org/abs/2006.10204)
