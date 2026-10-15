#!/usr/bin/env python3
"""
Diagnostic: Compare pose sequences between training data and live extraction.

This will help identify if there's a mismatch in how poses are being extracted.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load a training sequence
training_seq_path = Path("training_data/normal/processed")
if training_seq_path.exists():
    training_files = list(training_seq_path.glob("*.npy"))
    if training_files:
        training_seq = np.load(training_files[0])
        print("TRAINING DATA FORMAT:")
        print(f"  File: {training_files[0].name}")
        print(f"  Shape: {training_seq.shape}")
        print(f"  Dtype: {training_seq.dtype}")
        print(f"  Value range: [{training_seq.min():.6f}, {training_seq.max():.6f}]")
        print(f"  Mean: {training_seq.mean():.6f}")
        print(f"  Std: {training_seq.std():.6f}")
        print("\n  First landmark (frame 0, keypoint 0):")
        print(f"    x={training_seq[0,0,0]:.6f}, y={training_seq[0,0,1]:.6f}, z={training_seq[0,0,2]:.6f}")
        print("\n  Sample values (frame 0, first 3 keypoints):")
        print(training_seq[0, :3, :])

# Now test live extraction
print("\n" + "="*60)
print("LIVE EXTRACTION TEST:")
print("="*60)

from seizure_detection.analysis import pose_detection
from seizure_detection.analysis.onnx_detector import ONNXSeizureDetector
import cv2
from datetime import datetime

# Initialize
pose_detection.setup()
detector = ONNXSeizureDetector(
    model_path=Path("models/hypermotor_detector.onnx"),
    sequence_length=30
)

# Load a test video
test_video = Path("training_data/test/normal/video_20250107_133603.mp4")
if not test_video.exists():
    # Try alternative location
    test_video = Path("training_data/normal/videos/video_20250107_133603.mp4")

if test_video.exists():
    cap = cv2.VideoCapture(str(test_video))
    
    frame_count = 0
    base_timestamp = datetime.now()
    
    # Extract 30 frames
    while frame_count < 100:  # Get some frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = base_timestamp.replace(second=(base_timestamp.second + frame_count) % 60)
        
        # Extract pose
        _, pose_telemetry = pose_detection.process_frame(frame, do_draw=False)
        pose_data = pose_telemetry.get("pose", {})
        
        if pose_data:
            landmarks = pose_data.get("landmarks", [])
            if landmarks:
                detector.add_pose(landmarks, timestamp)
                
                if frame_count == 1:
                    # Show first extracted pose
                    print("\n  Frame 1 landmarks extracted:")
                    if len(landmarks) > 0 and isinstance(landmarks[0], list):
                        print(f"    Total landmarks: {len(landmarks[0])}")
                        print("    First landmark:")
                        lm = landmarks[0][0]
                        print(f"      x={lm.get('x', 0):.6f}, y={lm.get('y', 0):.6f}, z={lm.get('z', 0):.6f}")
                        print("\n    First 3 landmarks:")
                        for i, lm in enumerate(landmarks[0][:3]):
                            print(f"      {i}: x={lm.get('x', 0):.6f}, y={lm.get('y', 0):.6f}, z={lm.get('z', 0):.6f}")
                
                if len(detector.pose_buffer) == 30:
                    # Check the buffer
                    live_seq = np.array(list(detector.pose_buffer))
                    print("\n  LIVE BUFFER (after 30 frames):")
                    print(f"    Shape: {live_seq.shape}")
                    print(f"    Dtype: {live_seq.dtype}")
                    print(f"    Value range: [{live_seq.min():.6f}, {live_seq.max():.6f}]")
                    print(f"    Mean: {live_seq.mean():.6f}")
                    print(f"    Std: {live_seq.std():.6f}")
                    print("\n    First landmark (frame 0, keypoint 0):")
                    print(f"      x={live_seq[0,0,0]:.6f}, y={live_seq[0,0,1]:.6f}, z={live_seq[0,0,2]:.6f}")
                    print("\n    Sample values (frame 0, first 3 keypoints):")
                    print(live_seq[0, :3, :])
                    
                    # Run prediction
                    result = detector.predict()
                    if result and result.get('ready'):
                        print(f"\n    Model prediction: {result['probability']:.6f}")
                    
                    break
    
    cap.release()
    pose_detection.release()
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    if 'training_seq' in locals() and 'live_seq' in locals():
        print(f"Training mean: {training_seq.mean():.6f}")
        print(f"Live mean:     {live_seq.mean():.6f}")
        print(f"Difference:    {abs(training_seq.mean() - live_seq.mean()):.6f}")
else:
    print("Test video not found")
