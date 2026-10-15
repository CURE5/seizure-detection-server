#!/usr/bin/env python3
"""
Test seizure detection on specific videos to tune thresholds.

source sz/bin/activate && 
python scripts/test_detection_on_videos.py 
--video-dir training_data/test/seizure --output test_results_sz_both.json

"""
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis import MultiStageDetector
from seizure_detection.analysis.pose_detection import setup as setup_pose_detection, release as release_pose_detection


def analyze_video(video_path: Path, detector: MultiStageDetector):
    """Analyze a video file frame by frame."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {video_path.name}")
    print(f"{'='*80}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print("Video info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")
    print()
    
    # Detection tracking
    results = []
    frame_count = 0
    
    # Stage detections
    stage1_detections = 0  # Motion threshold
    stage2_detections = 0  # Activity detection  
    stage3_detections = 0  # Motion model
    stage4_detections = 0  # Pose model
    persistence_detections = 0  # Persistence detector (pose)
    motion_persistence_detections = 0  # Motion persistence detector
    
    # Confidence scores
    motion_confidences = []
    pose_confidences = []
    persistence_scores = []
    motion_persistence_scores = []
    
    # Start timestamp for monotonic time
    base_timestamp = datetime.now()
    
    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Use monotonically increasing timestamp (Unix timestamp as float)
        timestamp = base_timestamp.timestamp() + (frame_count / fps) if fps > 0 else base_timestamp.timestamp() + frame_count
        
        # Process through detector (it extracts poses internally)
        try:
            detection_result, _ = detector.analyze_frame(frame, timestamp, show_overlay=False)
        except Exception as e:
            if "monotonically increasing" in str(e):
                # Skip this frame
                continue
            else:
                print(f"\n  ⚠️  Error at frame {frame_count}: {e}")
                continue
        
        # Track detections by stage
        if detection_result.motion_detected:
            stage1_detections += 1
        if detection_result.activity_detected:
            stage2_detections += 1
        if detection_result.motion_model_detected:
            stage3_detections += 1
        if detection_result.pose_model_detected:
            stage4_detections += 1
            # Print live detections
            pose_conf = detection_result.pose_model_confidence or 0.0
            print(f"\n  🚨 Frame {frame_count:5d} ({frame_count/fps:6.1f}s): "
                  f"SEIZURE DETECTED (confidence: {pose_conf:.3f})")
        if detection_result.persistence_detected:
            persistence_detections += 1
            persist_score = detection_result.persistence_score or 0.0
            persist_dur = detection_result.persistence_duration or 0.0
            print(f"\n  ⚡ Frame {frame_count:5d} ({frame_count/fps:6.1f}s): "
                  f"SUSTAINED SEIZURE (persistence: {persist_score:.1%}, duration: {persist_dur:.1f}s)")
        if hasattr(detection_result, 'motion_persistence_detected') and detection_result.motion_persistence_detected:
            motion_persistence_detections += 1
            motion_persist_score = detection_result.motion_persistence_score or 0.0
            motion_persist_dur = detection_result.motion_persistence_duration or 0.0
            print(f"\n  💥 Frame {frame_count:5d} ({frame_count/fps:6.1f}s): "
                  f"SUSTAINED MOTION (persistence: {motion_persist_score:.1%}, duration: {motion_persist_dur:.1f}s)")
        
        # Debug: Show activity confidence when motion is detected but activity is not
        if detection_result.motion_detected and not detection_result.activity_detected and frame_count % 50 == 0:
            act_conf = detection_result.activity_confidence or 0.0
            print(f"\n  ⚠️  Frame {frame_count:5d}: Motion detected but Activity blocked (conf: {act_conf:.3f}, threshold: {detection_result.activity_threshold})")
        
        # Track confidence scores
        if detection_result.motion_model_confidence is not None:
            motion_confidences.append(detection_result.motion_model_confidence)
        if detection_result.pose_model_confidence is not None:
            pose_confidences.append(detection_result.pose_model_confidence)
        if detection_result.persistence_score is not None:
            persistence_scores.append(detection_result.persistence_score)
        if hasattr(detection_result, 'motion_persistence_score') and detection_result.motion_persistence_score is not None:
            motion_persistence_scores.append(detection_result.motion_persistence_score)
        
        # Store result
        timestamp_sec = frame_count / fps if fps > 0 else frame_count
        result_dict = {
            'frame': int(frame_count),
            'timestamp': float(timestamp_sec),
            'motion_detected': bool(detection_result.motion_detected),
            'motion_degree': float(detection_result.motion_degree) if detection_result.motion_degree is not None else None,
            'activity_detected': bool(detection_result.activity_detected),
            'activity_confidence': float(detection_result.activity_confidence) if detection_result.activity_confidence is not None else None,
            'motion_model_detected': bool(detection_result.motion_model_detected),
            'motion_model_confidence': float(detection_result.motion_model_confidence) if detection_result.motion_model_confidence is not None else None,
            'pose_model_detected': bool(detection_result.pose_model_detected),
            'pose_model_confidence': float(detection_result.pose_model_confidence) if detection_result.pose_model_confidence is not None else None,
            'persistence_detected': bool(detection_result.persistence_detected),
            'persistence_score': float(detection_result.persistence_score) if detection_result.persistence_score is not None else None,
            'persistence_duration': float(detection_result.persistence_duration) if detection_result.persistence_duration is not None else None,
            'seizure_detected': bool(detection_result.seizure_onset or detection_result.persistence_detected or detection_result.pose_model_detected or detection_result.motion_model_detected),
        }
        
        # Add motion persistence fields if available
        if hasattr(detection_result, 'motion_persistence_detected'):
            result_dict['motion_persistence_detected'] = bool(detection_result.motion_persistence_detected)
            result_dict['motion_persistence_score'] = float(detection_result.motion_persistence_score) if detection_result.motion_persistence_score is not None else None
            result_dict['motion_persistence_duration'] = float(detection_result.motion_persistence_duration) if detection_result.motion_persistence_duration is not None else None
        
        results.append(result_dict)
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')
    
    cap.release()
    print()  # New line after progress
    
    # Calculate statistics
    print(f"\n{'─'*80}")
    print("DETECTION RESULTS:")
    print(f"{'─'*80}")
    print(f"Stage 1 (Motion threshold):      {stage1_detections:4d} / {frame_count} frames ({stage1_detections/frame_count*100:5.1f}%)")
    print(f"Stage 2 (Activity detection):    {stage2_detections:4d} / {frame_count} frames ({stage2_detections/frame_count*100:5.1f}%)")
    print(f"Stage 3 (Motion LSTM):           {stage3_detections:4d} / {frame_count} frames ({stage3_detections/frame_count*100:5.1f}%)")
    print(f"Stage 4 (Pose model):            {stage4_detections:4d} / {frame_count} frames ({stage4_detections/frame_count*100:5.1f}%)")
    print(f"Persistence (Sustained pose):    {persistence_detections:4d} / {frame_count} frames ({persistence_detections/frame_count*100:5.1f}%)")
    print(f"Motion Persistence (Sustained):  {motion_persistence_detections:4d} / {frame_count} frames ({motion_persistence_detections/frame_count*100:5.1f}%)")
    
    # Confidence statistics
    if motion_confidences:
        print(f"\n{'─'*80}")
        print("MOTION MODEL CONFIDENCE SCORES:")
        print(f"{'─'*80}")
        print(f"  Mean:   {np.mean(motion_confidences):.4f}")
        print(f"  Median: {np.median(motion_confidences):.4f}")
        print(f"  Max:    {np.max(motion_confidences):.4f}")
        print(f"  Min:    {np.min(motion_confidences):.4f}")
        print(f"  Std:    {np.std(motion_confidences):.4f}")
        print(f"  P95:    {np.percentile(motion_confidences, 95):.4f}")
        print(f"  P99:    {np.percentile(motion_confidences, 99):.4f}")
    
    if pose_confidences:
        print(f"\n{'─'*80}")
        print("POSE MODEL CONFIDENCE SCORES:")
        print(f"{'─'*80}")
        print(f"  Mean:   {np.mean(pose_confidences):.4f}")
        print(f"  Median: {np.median(pose_confidences):.4f}")
        print(f"  Max:    {np.max(pose_confidences):.4f}")
        print(f"  Min:    {np.min(pose_confidences):.4f}")
        print(f"  Std:    {np.std(pose_confidences):.4f}")
        print(f"  P95:    {np.percentile(pose_confidences, 95):.4f}")
        print(f"  P99:    {np.percentile(pose_confidences, 99):.4f}")
    
    if persistence_scores:
        print(f"\n{'─'*80}")
        print("PERSISTENCE SCORES (POSE):")
        print(f"{'─'*80}")
        print(f"  Mean:   {np.mean(persistence_scores):.4f}")
        print(f"  Median: {np.median(persistence_scores):.4f}")
        print(f"  Max:    {np.max(persistence_scores):.4f}")
        print(f"  Min:    {np.min(persistence_scores):.4f}")
        print(f"  Std:    {np.std(persistence_scores):.4f}")
        print(f"  P95:    {np.percentile(persistence_scores, 95):.4f}")
        print(f"  P99:    {np.percentile(persistence_scores, 99):.4f}")
    
    if motion_persistence_scores:
        print(f"\n{'─'*80}")
        print("MOTION PERSISTENCE SCORES:")
        print(f"{'─'*80}")
        print(f"  Mean:   {np.mean(motion_persistence_scores):.4f}")
        print(f"  Median: {np.median(motion_persistence_scores):.4f}")
        print(f"  Max:    {np.max(motion_persistence_scores):.4f}")
        print(f"  Min:    {np.min(motion_persistence_scores):.4f}")
        print(f"  Std:    {np.std(motion_persistence_scores):.4f}")
        print(f"  P95:    {np.percentile(motion_persistence_scores, 95):.4f}")
        print(f"  P99:    {np.percentile(motion_persistence_scores, 99):.4f}")
    
    # Seizure detection summary
    seizure_frames = sum(1 for r in results if r['seizure_detected'])
    print(f"\n{'─'*80}")
    print("SEIZURE DETECTION SUMMARY:")
    print(f"{'─'*80}")
    print(f"  Seizure detected in: {seizure_frames} / {frame_count} frames ({seizure_frames/frame_count*100:.1f}%)")
    
    return {
        'video': video_path.name,
        'frames': frame_count,
        'duration': duration,
        'fps': fps,
        'stage1_detections': stage1_detections,
        'stage2_detections': stage2_detections,
        'stage3_detections': stage3_detections,
        'stage4_detections': stage4_detections,
        'persistence_detections': persistence_detections,
        'motion_persistence_detections': motion_persistence_detections,
        'seizure_frames': seizure_frames,
        'motion_confidence_stats': {
            'mean': float(np.mean(motion_confidences)) if motion_confidences else None,
            'median': float(np.median(motion_confidences)) if motion_confidences else None,
            'max': float(np.max(motion_confidences)) if motion_confidences else None,
            'min': float(np.min(motion_confidences)) if motion_confidences else None,
            'p95': float(np.percentile(motion_confidences, 95)) if motion_confidences else None,
            'p99': float(np.percentile(motion_confidences, 99)) if motion_confidences else None,
        },
        'pose_confidence_stats': {
            'mean': float(np.mean(pose_confidences)) if pose_confidences else None,
            'median': float(np.median(pose_confidences)) if pose_confidences else None,
            'max': float(np.max(pose_confidences)) if pose_confidences else None,
            'min': float(np.min(pose_confidences)) if pose_confidences else None,
            'p95': float(np.percentile(pose_confidences, 95)) if pose_confidences else None,
            'p99': float(np.percentile(pose_confidences, 99)) if pose_confidences else None,
        },
        'persistence_stats': {
            'mean': float(np.mean(persistence_scores)) if persistence_scores else None,
            'median': float(np.median(persistence_scores)) if persistence_scores else None,
            'max': float(np.max(persistence_scores)) if persistence_scores else None,
            'min': float(np.min(persistence_scores)) if persistence_scores else None,
            'p95': float(np.percentile(persistence_scores, 95)) if persistence_scores else None,
            'p99': float(np.percentile(persistence_scores, 99)) if persistence_scores else None,
        },
        'motion_persistence_stats': {
            'mean': float(np.mean(motion_persistence_scores)) if motion_persistence_scores else None,
            'median': float(np.median(motion_persistence_scores)) if motion_persistence_scores else None,
            'max': float(np.max(motion_persistence_scores)) if motion_persistence_scores else None,
            'min': float(np.min(motion_persistence_scores)) if motion_persistence_scores else None,
            'p95': float(np.percentile(motion_persistence_scores, 95)) if motion_persistence_scores else None,
            'p99': float(np.percentile(motion_persistence_scores, 99)) if motion_persistence_scores else None,
        },
        'detailed_results': results
    }


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test seizure detection on videos')
    parser.add_argument('--video-dir', type=str, 
                       default='training_data/test/seizure',
                       help='Directory containing test videos')
    parser.add_argument('--motion-threshold', type=float, default=0.9,
                       help='Motion model threshold')
    parser.add_argument('--pose-threshold', type=float, default=0.5,
                       help='Pose model threshold')
    parser.add_argument('--activity-threshold', type=float, default=0.01,
                       help='Activity detection threshold (default 0.01 for testing, production uses 0.7)')
    parser.add_argument('--output', type=str, default='detection_test_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Initialize pose detection backend FIRST
    print("Initializing pose detection...")
    setup_pose_detection()
    
    # Initialize detector
    print("Initializing multi-stage detector...")
    print(f"  Motion model threshold: {args.motion_threshold}")
    print(f"  Pose model threshold: {args.pose_threshold}")
    print(f"  Activity threshold: {args.activity_threshold}")
    
    # Temporarily override ACTIVITY_THRESHOLD for testing
    import os
    os.environ['ACTIVITY_THRESHOLD'] = str(args.activity_threshold)
    
    detector = MultiStageDetector(
        motion_model_path=Path('models/motion_seizure_detector.onnx'),
        motion_model_threshold=args.motion_threshold,
        pose_model_path=Path('models/hypermotor_detector.onnx'),
        pose_model_threshold=args.pose_threshold,
        enable_time_weighting=True,
        intra_seizure_max_gap=30.0,
        seizure_end_threshold=60.0
    )
    
    # Find videos
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"ERROR: Video directory not found: {video_dir}")
        return
    
    video_files = list(video_dir.glob('*.mp4'))
    if not video_files:
        print(f"ERROR: No .mp4 files found in {video_dir}")
        return
    
    print(f"\nFound {len(video_files)} video(s) to analyze")
    
    # Analyze each video
    all_results = []
    for video_path in sorted(video_files):
        result = analyze_video(video_path, detector)
        if result:
            all_results.append(result)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'motion_threshold': args.motion_threshold,
            'pose_threshold': args.pose_threshold,
            'videos': all_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
    
    # Recommendations
    print("\n" + "="*80)
    print("THRESHOLD RECOMMENDATIONS:")
    print("="*80)
    
    for result in all_results:
        print(f"\n{result['video']}:")
        
        if result['motion_confidence_stats']['max']:
            motion_p95 = result['motion_confidence_stats']['p95']
            motion_max = result['motion_confidence_stats']['max']
            print("  Motion model:")
            print(f"    Current threshold: {args.motion_threshold:.3f}")
            print(f"    P95 confidence: {motion_p95:.3f}")
            print(f"    Max confidence: {motion_max:.3f}")
            
            if motion_max < args.motion_threshold:
                print(f"    ⚠️  WARNING: Max confidence ({motion_max:.3f}) < threshold ({args.motion_threshold:.3f})")
                print(f"    💡 Recommend lowering to: {motion_p95:.3f}")
            else:
                print("    ✓ Threshold is appropriate")
        
        if result['pose_confidence_stats']['max']:
            pose_p95 = result['pose_confidence_stats']['p95']
            pose_max = result['pose_confidence_stats']['max']
            print("  Pose model:")
            print(f"    Current threshold: {args.pose_threshold:.3f}")
            print(f"    P95 confidence: {pose_p95:.3f}")
            print(f"    Max confidence: {pose_max:.3f}")
            
            if pose_max < args.pose_threshold:
                print(f"    ⚠️  WARNING: Max confidence ({pose_max:.3f}) < threshold ({args.pose_threshold:.3f})")
                print(f"    💡 Recommend lowering to: {pose_p95:.3f}")
            else:
                print("    ✓ Threshold is appropriate")
    
    # Cleanup
    release_pose_detection()


if __name__ == '__main__':
    main()
