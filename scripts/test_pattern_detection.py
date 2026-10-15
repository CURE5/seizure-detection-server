#!/usr/bin/env python3
"""
Test rhythmic pattern detector on seizure videos.

Compares:
1. Single-frame pose model predictions (current approach)
2. Rhythmic pattern detection over 60s window (new approach)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis import MultiStageDetector
from seizure_detection.config import SeizureDetectionConfig


def test_pattern_detection(video_path: Path):
    """Test pattern detection on a video."""
    print(f"\n{'='*80}")
    print(f"Testing: {video_path.name}")
    print(f"{'='*80}")
    
    # Initialize detector
    cfg = SeizureDetectionConfig()
    cfg.MOTION_THRESHOLD = 25
    cfg.MOTION_DEGREE_THRESHOLD = 0.003
    cfg.ACTIVITY_THRESHOLD = 0.0
    cfg.USE_POSE_MODEL = True
    cfg.USE_MOTION_MODEL = False
    
    detector = MultiStageDetector(
        config=cfg,
        pose_model_path=Path("models/hypermotor_detector.onnx"),
        pose_model_threshold=0.5,
        intra_seizure_max_gap=30.0,
        seizure_end_threshold=60.0
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Could not open: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"FPS: {fps}, Frames: {total_frames}, Duration: {duration:.1f}s\n")
    
    # Track results
    frame_detections = []  # Single-frame pose model detections
    pattern_detections = []  # Rhythmic pattern detections
    frame_count = 0
    
    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = datetime.now(timezone.utc) + timedelta(seconds=frame_count / fps)
        result, _ = detector.analyze_frame(frame, timestamp, show_overlay=False)
        
        time_sec = frame_count / fps
        
        # Track single-frame detections
        if result.pose_model_detected:
            frame_detections.append({
                'frame': frame_count,
                'time': time_sec,
                'confidence': result.pose_model_confidence
            })
            print(f"  Frame {frame_count:4d} ({time_sec:5.1f}s): pose={result.pose_model_confidence:.3f} ⚡")
        
        # Track pattern detections
        if result.rhythmic_pattern_detected:
            pattern_detections.append({
                'frame': frame_count,
                'time': time_sec,
                'confidence': result.rhythmic_pattern_confidence,
                'spasm_count': result.rhythmic_spasm_count,
                'avg_interval': result.rhythmic_avg_interval
            })
            print(f"  Frame {frame_count:4d} ({time_sec:5.1f}s): PATTERN detected! "
                  f"spasms={result.rhythmic_spasm_count}, "
                  f"interval={result.rhythmic_avg_interval:.1f}s, "
                  f"confidence={result.rhythmic_pattern_confidence:.3f} 🔄")
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  ... processed {frame_count}/{total_frames} frames")
    
    cap.release()
    
    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    frame_rate = len(frame_detections) / total_frames * 100 if total_frames > 0 else 0
    pattern_rate = len(pattern_detections) / total_frames * 100 if total_frames > 0 else 0
    
    print("\nSingle-Frame Pose Model:")
    print(f"  Detections: {len(frame_detections)}/{total_frames} frames ({frame_rate:.1f}%)")
    if frame_detections:
        confs = [d['confidence'] for d in frame_detections]
        print(f"  Confidence range: {min(confs):.3f} - {max(confs):.3f}")
        print(f"  Mean confidence: {sum(confs)/len(confs):.3f}")
        print(f"  Detection window: {frame_detections[0]['time']:.1f}s - {frame_detections[-1]['time']:.1f}s")
    
    print("\nRhythmic Pattern Detection:")
    print(f"  Detections: {len(pattern_detections)}/{total_frames} frames ({pattern_rate:.1f}%)")
    if pattern_detections:
        # Find unique pattern events (group consecutive detections)
        pattern_events = []
        current_event = None
        
        for p in pattern_detections:
            if current_event is None:
                current_event = p
            elif p['frame'] - pattern_events[-1]['frame'] > 5:  # New event if >1s gap
                pattern_events.append(current_event)
                current_event = p
        
        if current_event:
            pattern_events.append(current_event)
        
        print(f"  Unique pattern events: {len(pattern_events)}")
        for i, event in enumerate(pattern_events, 1):
            print(f"    Event {i}: {event['time']:.1f}s, "
                  f"{event['spasm_count']} spasms, "
                  f"interval={event['avg_interval']:.1f}s")
    
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    if frame_rate > 20:
        print(f"⚠️  HIGH single-frame detection rate ({frame_rate:.1f}%)")
        print("   This suggests possible false positives from sliding window")
    
    if pattern_rate > 0:
        print("✓ Pattern detector found rhythmic spasms")
        print("  This is stronger evidence of hypermotor seizure")
    else:
        print("⚠️  No rhythmic pattern detected")
        if frame_rate > 0:
            print("  Pose model detected motion, but not in rhythmic pattern")
        else:
            print("  No seizure detected")
    
    return {
        'frame_detections': len(frame_detections),
        'pattern_detections': len(pattern_detections),
        'frame_rate': frame_rate,
        'pattern_rate': pattern_rate
    }


def main():
    """Test pattern detector on seizure and normal videos."""
    print("="*80)
    print("RHYTHMIC PATTERN DETECTION TESTING")
    print("="*80)
    print("\nTesting new pattern detector that analyzes 60s of pose predictions")
    print("to find characteristic spasm→calm→spasm rhythm of hypermotor seizures\n")
    
    training_data_dir = Path(__file__).parent.parent / "training_data"
    
    # Test on seizure videos
    seizure_dir = training_data_dir / "test" / "seizure"
    if seizure_dir.exists():
        seizure_videos = list(seizure_dir.glob("*.mp4"))[:2]
        print(f"\nTesting {len(seizure_videos)} SEIZURE videos:\n")
        
        for video in seizure_videos:
            test_pattern_detection(video)
    
    # Test on normal videos
    normal_dir = training_data_dir / "test" / "normal"
    if normal_dir.exists():
        normal_videos = list(normal_dir.glob("*.mp4"))[:1]
        print(f"\n\nTesting {len(normal_videos)} NORMAL video:\n")
        
        for video in normal_videos:
            result = test_pattern_detection(video)
            
            if result and result['pattern_rate'] > 0:
                print("\n❌ FALSE POSITIVE: Pattern detected in normal video!")
                print("   This suggests pattern detector thresholds may be too low")


if __name__ == "__main__":
    main()
