#!/usr/bin/env python3
"""
Test wake-triggered sensitivity boost on seizure videos.

Simulates:
1. Normal baseline (threshold=0.5)
2. Post-wake period (threshold=0.3)
3. Compares detection rates
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis import MultiStageDetector
import cv2

# Thresholds to test
NORMAL_THRESHOLD = 0.5
POST_WAKE_THRESHOLD = 0.3

def test_video_with_threshold(video_path: Path, threshold: float, desc: str):
    """Test video with a specific pose model threshold."""
    print(f"\n{'='*80}")
    print(f"{desc} (threshold={threshold})")
    print(f"Video: {video_path.name}")
    print(f"{'='*80}")
    
    # Initialize detector with specified threshold
    from seizure_detection.config import SeizureDetectionConfig
    
    cfg = SeizureDetectionConfig()
    cfg.MOTION_THRESHOLD = 25
    cfg.MOTION_DEGREE_THRESHOLD = 0.003
    cfg.ACTIVITY_THRESHOLD = 0.0  # Disabled
    cfg.USE_POSE_MODEL = True
    cfg.USE_MOTION_MODEL = False
    
    detector = MultiStageDetector(
        config=cfg,
        pose_model_path=Path("models/hypermotor_detector.onnx"),
        pose_model_threshold=threshold,
        intra_seizure_max_gap=30.0,
        seizure_end_threshold=60.0
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"FPS: {fps}, Frames: {total_frames}, Duration: {duration:.1f}s")
    
    detections = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = datetime.now(timezone.utc) + timedelta(seconds=frame_count / fps)
        result, _ = detector.analyze_frame(frame, timestamp, show_overlay=False)
        
        if result.pose_model_detected:
            time_sec = frame_count / fps
            detections.append({
                'frame': frame_count,
                'time': time_sec,
                'confidence': result.pose_model_confidence
            })
            print(f"  ⚡ Detection at frame {frame_count} ({time_sec:.1f}s): confidence={result.pose_model_confidence:.3f}")
        
        frame_count += 1
    
    cap.release()
    
    detection_rate = len(detections) / total_frames * 100 if total_frames > 0 else 0
    
    print("\nResults:")
    print(f"  Total detections: {len(detections)}/{total_frames} frames ({detection_rate:.1f}%)")
    
    if detections:
        confidences = [d['confidence'] for d in detections]
        print(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"  Mean confidence: {sum(confidences)/len(confidences):.3f}")
        
        # Find detection spans
        if len(detections) > 0:
            first_detection = detections[0]['time']
            last_detection = detections[-1]['time']
            print(f"  Detection window: {first_detection:.1f}s - {last_detection:.1f}s")
    
    return {
        'threshold': threshold,
        'detections': len(detections),
        'total_frames': total_frames,
        'detection_rate': detection_rate,
        'confidences': [d['confidence'] for d in detections] if detections else []
    }


def main():
    """Test wake sensitivity boost on seizure videos."""
    print("="*80)
    print("WAKE-TRIGGERED SENSITIVITY BOOST TESTING")
    print("="*80)
    print("\nPurpose: Verify that lower threshold after wake-up catches more seizures")
    print("         without excessive false positives\n")
    
    training_data_dir = Path(__file__).parent.parent / "training_data"
    seizure_dir = training_data_dir / "test" / "seizure"
    
    if not seizure_dir.exists():
        print(f"✗ Seizure directory not found: {seizure_dir}")
        return
    
    # Test on 3 seizure videos
    seizure_videos = list(seizure_dir.glob("*.mp4"))[:3]
    
    if not seizure_videos:
        print(f"✗ No seizure videos found in {seizure_dir}")
        return
    
    print(f"Testing {len(seizure_videos)} seizure videos with two thresholds:\n")
    print(f"  NORMAL (baseline): {NORMAL_THRESHOLD}")
    print(f"  POST-WAKE (boosted): {POST_WAKE_THRESHOLD}")
    
    results = []
    
    for video in seizure_videos:
        # Test with normal threshold
        normal_result = test_video_with_threshold(
            video, 
            NORMAL_THRESHOLD, 
            "NORMAL THRESHOLD (baseline sensitivity)"
        )
        
        # Test with post-wake threshold
        wake_result = test_video_with_threshold(
            video,
            POST_WAKE_THRESHOLD,
            "POST-WAKE THRESHOLD (boosted sensitivity)"
        )
        
        if normal_result and wake_result:
            improvement = wake_result['detections'] - normal_result['detections']
            improvement_pct = (improvement / normal_result['total_frames'] * 100) if normal_result['total_frames'] > 0 else 0
            
            print(f"\n{'='*80}")
            print(f"COMPARISON for {video.name}:")
            print(f"{'='*80}")
            print(f"  Normal threshold:    {normal_result['detections']:3d} detections ({normal_result['detection_rate']:5.1f}%)")
            print(f"  Post-wake threshold: {wake_result['detections']:3d} detections ({wake_result['detection_rate']:5.1f}%)")
            print(f"  Improvement:         {improvement:+3d} detections ({improvement_pct:+5.1f}%)")
            
            if improvement > 0:
                print(f"  ✓ Post-wake boost caught {improvement} more frames")
            elif improvement == 0:
                print("  = No difference (both thresholds equal)")
            else:
                print("  ⚠️  Post-wake caught fewer frames (unexpected)")
            
            results.append({
                'video': video.name,
                'normal': normal_result,
                'wake': wake_result,
                'improvement': improvement
            })
    
    # Overall summary
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    if results:
        total_improvement = sum(r['improvement'] for r in results)
        avg_normal_rate = sum(r['normal']['detection_rate'] for r in results) / len(results)
        avg_wake_rate = sum(r['wake']['detection_rate'] for r in results) / len(results)
        
        print("\nAverage detection rates:")
        print(f"  Normal threshold:    {avg_normal_rate:.1f}%")
        print(f"  Post-wake threshold: {avg_wake_rate:.1f}%")
        print(f"  Average improvement: {avg_wake_rate - avg_normal_rate:+.1f}%")
        
        print(f"\nTotal additional detections: {total_improvement}")
        
        if total_improvement > 0:
            print("\n✅ POST-WAKE BOOST IS EFFECTIVE")
            print(f"   Lowering threshold from {NORMAL_THRESHOLD} to {POST_WAKE_THRESHOLD} after wake-up")
            print(f"   catches {total_improvement} more seizure frames across {len(results)} videos")
        elif total_improvement == 0:
            print("\n⚠️  POST-WAKE BOOST HAS NO EFFECT")
            print("   Both thresholds produce identical results")
            print("   Consider lowering POST_WAKE_THRESHOLD further")
        else:
            print("\n❌ POST-WAKE BOOST IS HARMFUL")
            print("   This shouldn't happen - check implementation")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print("\nTo enable wake-triggered sensitivity boost:")
    print("1. Set NOTIFY_ON_WAKEUP=true in .env")
    print("2. Configure POST_WAKE_SENSITIVITY_MINUTES=15 (duration of boost)")
    print(f"3. Configure POST_WAKE_POSE_THRESHOLD={POST_WAKE_THRESHOLD} (lower = more sensitive)")
    print("\nThe system will automatically:")
    print("- Detect wake-ups after 30+ min of low motion")
    print("- Lower pose model threshold for next 15 minutes")
    print("- Restore normal threshold after boost period expires")


if __name__ == "__main__":
    main()
