#!/usr/bin/env python3
"""
Test pose model directly on videos WITHOUT motion gates.

This validates that the pose model itself works correctly,
independent of the multi-stage pipeline gating logic.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis import pose_detection
from seizure_detection.analysis.onnx_detector import ONNXSeizureDetector


def test_video_with_pose_model(
    video_path: Path,
    model_path: Path,
    threshold: float = 0.5
):
    """
    Test pose model directly on video.
    
    Bypasses all motion gating - just extracts poses and runs model.
    """
    print(f"\n{'='*70}")
    print(f"Testing: {video_path.name}")
    print(f"{'='*70}")
    
    # Initialize pose detection
    print("Initializing MediaPipe...")
    pose_detection.setup()
    
    # Initialize pose model
    print("Loading ONNX model...")
    detector = ONNXSeizureDetector(
        model_path=model_path,
        sequence_length=30,
        confidence_threshold=threshold
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print("Video info:")
    print(f"  FPS: {fps:.1f}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {duration:.1f}s")
    print()
    
    # Process frames
    frame_idx = 0
    poses_extracted = 0
    detections = []
    detection_frames = []
    
    # Start with base timestamp and increment
    base_timestamp = datetime.now()
    
    print("Processing frames...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use monotonically increasing timestamp (frame number based)
            timestamp = base_timestamp.replace(microsecond=frame_idx * 1000)
            
            # Extract pose
            try:
                _, pose_telemetry = pose_detection.process_frame(frame, do_draw=False)
                pose_data = pose_telemetry.get("pose", {})
                
                landmarks = None
                if pose_data:
                    landmarks = pose_data.get("landmarks", [])
                
                # Add to pose model
                if landmarks:
                    poses_extracted += 1
                    detector.add_pose(landmarks, timestamp)
                    
                    # Check if model is ready to predict
                    if detector.is_ready():
                        result = detector.predict()
                        
                        if result and result.get('ready'):
                            confidence = result['probability']
                            detected = confidence >= threshold
                            
                            detections.append({
                                'frame': frame_idx,
                                'timestamp': frame_idx / fps if fps > 0 else 0,
                                'confidence': confidence,
                                'detected': detected
                            })
                            
                            if detected:
                                detection_frames.append(frame_idx)
                                print(f"  🚨 Frame {frame_idx:5d} ({frame_idx/fps:6.1f}s): "
                                      f"SEIZURE DETECTED (confidence: {confidence:.3f})")
            except Exception as e:
                if "monotonically increasing" in str(e):
                    # Skip this frame, timestamp issue
                    pass
                else:
                    print(f"  ⚠️  Error at frame {frame_idx}: {e}")
            
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 100 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"  Progress: {progress:5.1f}% - Poses: {poses_extracted:4d}", end='\r')
    
    finally:
        cap.release()
        pose_detection.release()
    
    print()  # New line after progress
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Frames processed: {frame_idx}")
    print(f"Poses extracted: {poses_extracted} ({poses_extracted/frame_idx*100:.1f}%)")
    print(f"Model predictions: {len(detections)}")
    print(f"Seizure detections: {len(detection_frames)}")
    
    if detection_frames:
        print(f"\nDetection frames: {detection_frames[:10]}{'...' if len(detection_frames) > 10 else ''}")
        print(f"First detection: frame {detection_frames[0]} ({detection_frames[0]/fps:.1f}s)")
        print(f"Last detection: frame {detection_frames[-1]} ({detection_frames[-1]/fps:.1f}s)")
        
        # Calculate detection statistics
        confidences = [d['confidence'] for d in detections if d['detected']]
        if confidences:
            print("\nConfidence statistics:")
            print(f"  Min: {min(confidences):.3f}")
            print(f"  Max: {max(confidences):.3f}")
            print(f"  Mean: {np.mean(confidences):.3f}")
    else:
        print("\n❌ NO SEIZURES DETECTED")
        
        # Show max confidence
        if detections:
            max_conf = max(d['confidence'] for d in detections)
            print(f"   Max confidence: {max_conf:.3f} (threshold: {threshold})")
    
    print(f"{'='*70}\n")
    
    return {
        'frames_processed': frame_idx,
        'poses_extracted': poses_extracted,
        'detections': len(detection_frames),
        'detection_frames': detection_frames
    }


def main():
    # Test videos
    test_dir = Path("training_data/test/seizure")
    test_videos = [
        test_dir / "video_20251209_115142.mp4",
        test_dir / "video_20251211_082502.mp4"
    ]
    
    # Model
    model_path = Path("models/hypermotor_detector.onnx")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python scripts/export_to_onnx.py")
        return
    
    print("\n🧠 Pose Model Direct Test")
    print(f"Model: {model_path}")
    print("Threshold: 0.5")
    
    # Test each video
    results = {}
    for video_path in test_videos:
        if video_path.exists():
            results[video_path.name] = test_video_with_pose_model(
                video_path, model_path, threshold=0.5
            )
        else:
            print(f"⚠️  Video not found: {video_path}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Poses: {result['poses_extracted']}/{result['frames_processed']} "
              f"({result['poses_extracted']/result['frames_processed']*100:.1f}%)")
        print(f"  Detections: {result['detections']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
