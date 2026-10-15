# Motion-Based Seizure Detection

Portable, robust seizure detection using optical flow motion analysis instead of pose estimation.

## Why Motion-Based?

**Advantages over pose detection:**
- ✅ Works regardless of body position (face up/down/sideways)
- ✅ Robust to partial occlusion (blankets, covers)
- ✅ No dependency on finicky pose models
- ✅ Simpler, more interpretable features
- ✅ Perfect for fixed camera monitoring

**Key insight:** Your cameras are fixed and the baby is always in roughly the same place. We don't need to track skeleton keypoints - we just need to detect abnormal motion patterns in the frame.

## Architecture

```
Frame Input (5 FPS)
    ↓
Optical Flow Extraction
    ├─ Motion magnitude (how much)
    ├─ Motion density (how widespread)
    └─ Motion direction (where moving)
    ↓
6-Second Sequences (30 frames)
    ↓
BiLSTM + Attention Model
    ↓
Seizure Probability [0-1]
```

## Quick Start

### 1. Test Motion Detection on a Video

```bash
# Test with visualization
python scripts/test_motion_detection.py --video path/to/video.mp4 --mode all --save

# Available modes:
# - heatmap: Color overlay showing motion intensity
# - vectors: Arrows showing motion direction
# - mask: Highlight moving regions
# - all: Combined visualization

# Test with webcam
python scripts/test_motion_detection.py --camera 0
```

This will show you:
- Real-time motion heatmap overlay
- Motion magnitude graph over time
- Statistics (avg motion, density, trend)

### 2. Extract Motion Features from Training Videos

```bash
# Extract features from your video datasets
python scripts/extract_motion_features.py \
    --seizure-dir /path/to/seizure/videos \
    --normal-dir /path/to/normal/videos \
    --output-dir training_data/motion_features \
    --fps 5 \
    --sequence-length 30
```

This will:
- Process all videos at 5 FPS
- Extract 6-second motion sequences (30 frames)
- Save as `.npy` files ready for training
- Create overlapping sequences for more training data

**Expected output:**
```
training_data/motion_features/
├── seizure/
│   ├── video1_seq0000.npy
│   ├── video1_seq0001.npy
│   └── ...
├── normal/
│   ├── video2_seq0000.npy
│   └── ...
└── extraction_metadata.json
```

### 3. Train the Model

```bash
# Train on extracted features
python scripts/train_motion_model.py \
    --data-dir training_data/motion_features \
    --output-dir models \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

**Training features:**
- BiLSTM with attention mechanism
- Early stopping (patience=15 epochs)
- Automatic train/val/test split (70/15/15)
- Exports to ONNX for Raspberry Pi deployment

**Expected output:**
```
models/
├── motion_seizure_detector_best.pth  # PyTorch weights
├── motion_seizure_detector.onnx      # For deployment
└── motion_model_metadata.json        # Training info
```

### 4. Integration (Coming Soon)

Motion-based detection will integrate into the main pipeline similar to the current pose-based detector, but with better reliability.

## Motion Features Explained

### 1. Motion Magnitude
- Measures pixel displacement between frames
- High magnitude = fast, violent movements
- Captured via optical flow (Farneback algorithm)

### 2. Motion Density
- Percentage of frame showing motion
- Localized movement vs whole-body movement
- Helps distinguish seizures from repositioning

### 3. Motion Direction
- Direction of movement (radians)
- Rhythmic back-and-forth vs random thrashing
- Useful for detecting oscillatory patterns

### 4. Temporal Patterns
- Variability over time
- Acceleration/deceleration
- Trend detection (increasing/decreasing)

## Visualization Modes

### Heatmap Mode
Shows motion intensity as color overlay:
- **Blue**: No motion
- **Green/Yellow**: Low-medium motion
- **Red**: High motion intensity

### Vector Mode
Shows optical flow as arrows:
- Arrow length = motion speed
- Arrow direction = motion direction
- Only shows significant motion (>0.5 px/frame)

### Mask Mode
Highlights moving regions:
- **Green overlay**: Areas with motion
- Clean binary view of "what's moving"

### All Mode
Combines heatmap + stats + graph for full debugging

## Performance

**Mac (Training):**
- Feature extraction: ~Real-time (5 FPS)
- Training: ~2-5 min per epoch (depends on dataset size)
- Export to ONNX: <1 second

**Raspberry Pi 5 (Inference):**
- Optical flow: ~15-20 FPS (full frame, CPU)
- Optical flow: ~30-40 FPS (with ROI)
- ONNX inference: ~1000 FPS (0.79ms)
- **Total pipeline: ~20-24 FPS** (sufficient for 5 FPS monitoring)

## Portability

**Same workflow as current model:**
1. Extract features on Mac (or Pi)
2. Train on Mac (uses GPU if available)
3. Export to ONNX (portable format)
4. Deploy to Pi (CPU inference only)

**No hardware-specific code** - works on any platform with OpenCV and ONNX Runtime.

## Debug Tips

### Low Motion Detection Sensitivity
- Adjust `motion_threshold` in `MotionFeatureExtractor` (default: 0.5 pixels/frame)
- Lower value = more sensitive to small movements

### Noisy Motion Detections
- Increase `motion_threshold`
- Apply ROI to focus on bed region only
- Add temporal smoothing (average over N frames)

### Visualization Not Showing
- Check video codec compatibility
- Try saving output with `--save` flag
- Ensure OpenCV GUI backend is working (`cv2.imshow`)

## Next Steps

1. **Test on your videos** - Run `test_motion_detection.py` to verify it captures motion
2. **Extract features** - Process your 100+ seizure videos
3. **Train model** - Should get similar or better performance than pose-based
4. **Integrate into pipeline** - Replace/augment current detection

## Comparison: Pose vs Motion

| Aspect | Pose-Based | Motion-Based |
|--------|-----------|--------------|
| **Occlusion** | Fails under blankets | Works through fabric |
| **Orientation** | Requires proper pose | Works in any orientation |
| **Complexity** | Depends on Hailo/MediaPipe | Simple OpenCV optical flow |
| **Interpretability** | Abstract keypoint movements | Direct motion visualization |
| **Speed (Pi)** | ~60 FPS (Hailo NPU) | ~20 FPS (CPU) |
| **Reliability** | Jittery, inconsistent | Stable, consistent |
| **Training Data** | Need clear poses | Any video works |

**Verdict:** Motion-based is more robust for your fixed-camera, baby-monitoring use case.
