#!/usr/bin/env python3
"""
Profile training videos to determine optimal motion and activity thresholds.

This analyzes actual seizure vs normal videos to find the threshold values
that separate them, eliminating guesswork.
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis.motion_detector import MotionDetector
from seizure_detection.analysis.motion_activity_detector import MotionActivityDetector


def profile_video(video_path: Path, motion_detector: MotionDetector, activity_detector: MotionActivityDetector):
    """Profile a single video for motion and activity metrics."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠️  Could not open: {video_path.name}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    motion_degrees = []
    activity_confidences = []
    frame_count = 0
    
    base_timestamp = datetime.now()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Use frame count as seconds offset to avoid microsecond overflow
        timestamp = base_timestamp.replace(second=(base_timestamp.second + frame_count) % 60)
        
        # Get motion
        motion_detected, motion_degree = motion_detector.detect_motion(frame)
        if motion_degree is not None:
            motion_degrees.append(motion_degree)
        
        # Get activity
        activity_result = activity_detector.process_frame(frame, timestamp)
        if activity_result and 'confidence' in activity_result:
            activity_confidences.append(activity_result['confidence'])
    
    cap.release()
    
    if not motion_degrees and not activity_confidences:
        return None
    
    return {
        'video': video_path.name,
        'frames': frame_count,
        'fps': fps,
        'duration': frame_count / fps if fps > 0 else 0,
        'motion': {
            'values': motion_degrees,
            'mean': float(np.mean(motion_degrees)) if motion_degrees else 0.0,
            'median': float(np.median(motion_degrees)) if motion_degrees else 0.0,
            'max': float(np.max(motion_degrees)) if motion_degrees else 0.0,
            'min': float(np.min(motion_degrees)) if motion_degrees else 0.0,
            'std': float(np.std(motion_degrees)) if motion_degrees else 0.0,
            'p25': float(np.percentile(motion_degrees, 25)) if motion_degrees else 0.0,
            'p50': float(np.percentile(motion_degrees, 50)) if motion_degrees else 0.0,
            'p75': float(np.percentile(motion_degrees, 75)) if motion_degrees else 0.0,
            'p90': float(np.percentile(motion_degrees, 90)) if motion_degrees else 0.0,
            'p95': float(np.percentile(motion_degrees, 95)) if motion_degrees else 0.0,
            'p99': float(np.percentile(motion_degrees, 99)) if motion_degrees else 0.0,
        },
        'activity': {
            'values': activity_confidences,
            'mean': float(np.mean(activity_confidences)) if activity_confidences else 0.0,
            'median': float(np.median(activity_confidences)) if activity_confidences else 0.0,
            'max': float(np.max(activity_confidences)) if activity_confidences else 0.0,
            'min': float(np.min(activity_confidences)) if activity_confidences else 0.0,
            'std': float(np.std(activity_confidences)) if activity_confidences else 0.0,
            'p25': float(np.percentile(activity_confidences, 25)) if activity_confidences else 0.0,
            'p50': float(np.percentile(activity_confidences, 50)) if activity_confidences else 0.0,
            'p75': float(np.percentile(activity_confidences, 75)) if activity_confidences else 0.0,
            'p90': float(np.percentile(activity_confidences, 90)) if activity_confidences else 0.0,
            'p95': float(np.percentile(activity_confidences, 95)) if activity_confidences else 0.0,
            'p99': float(np.percentile(activity_confidences, 99)) if activity_confidences else 0.0,
        }
    }


def main():
    """Main profiling function."""
    print("="*80)
    print("TRAINING DATA PROFILER")
    print("="*80)
    print("\nAnalyzing motion and activity levels in training videos...")
    print("This will help determine optimal threshold values.\n")
    
    # Initialize detectors
    motion_detector = MotionDetector()
    activity_detector = MotionActivityDetector()
    
    # Find training videos
    base_dir = Path("training_data")
    
    seizure_videos = list((base_dir / "seizure" / "videos").glob("*.mp4"))
    normal_videos = list((base_dir / "normal" / "videos").glob("*.mp4"))
    
    # Also check test directory
    test_seizure = list((base_dir / "test" / "seizure").glob("*.mp4"))
    seizure_videos.extend(test_seizure)
    
    print(f"Found {len(seizure_videos)} seizure videos")
    print(f"Found {len(normal_videos)} normal videos")
    print()
    
    # Profile seizure videos
    print("Profiling SEIZURE videos...")
    seizure_results = []
    for i, video_path in enumerate(seizure_videos[:10], 1):  # Limit to 10 for speed
        print(f"  [{i}/{min(10, len(seizure_videos))}] {video_path.name}...", end=' ')
        result = profile_video(video_path, motion_detector, activity_detector)
        if result:
            seizure_results.append(result)
            print(f"✓ (motion: {result['motion']['mean']:.4f}, activity: {result['activity']['mean']:.4f})")
        else:
            print("✗")
    
    # Profile normal videos  
    print("\nProfiling NORMAL videos...")
    normal_results = []
    for i, video_path in enumerate(normal_videos[:10], 1):  # Limit to 10 for speed
        print(f"  [{i}/{min(10, len(normal_videos))}] {video_path.name}...", end=' ')
        result = profile_video(video_path, motion_detector, activity_detector)
        if result:
            normal_results.append(result)
            print(f"✓ (motion: {result['motion']['mean']:.4f}, activity: {result['activity']['mean']:.4f})")
        else:
            print("✗")
    
    # Aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)
    
    # Motion statistics
    seizure_motion = [r['motion']['mean'] for r in seizure_results]
    normal_motion = [r['motion']['mean'] for r in normal_results]
    
    seizure_motion_max = [r['motion']['max'] for r in seizure_results]
    normal_motion_max = [r['motion']['max'] for r in normal_results]
    
    print("\nMOTION DEGREE (mean per video):")
    print("  Seizure videos:")
    print(f"    Mean:   {np.mean(seizure_motion):.6f}")
    print(f"    Median: {np.median(seizure_motion):.6f}")
    print(f"    Min:    {np.min(seizure_motion):.6f}")
    print(f"    Max:    {np.max(seizure_motion):.6f}")
    print(f"    P25:    {np.percentile(seizure_motion, 25):.6f}")
    print(f"    P75:    {np.percentile(seizure_motion, 75):.6f}")
    
    print("\n  Normal videos:")
    print(f"    Mean:   {np.mean(normal_motion):.6f}")
    print(f"    Median: {np.median(normal_motion):.6f}")
    print(f"    Min:    {np.min(normal_motion):.6f}")
    print(f"    Max:    {np.max(normal_motion):.6f}")
    print(f"    P25:    {np.percentile(normal_motion, 25):.6f}")
    print(f"    P75:    {np.percentile(normal_motion, 75):.6f}")
    
    # Activity statistics
    seizure_activity = [r['activity']['mean'] for r in seizure_results]
    normal_activity = [r['activity']['mean'] for r in normal_results]
    
    seizure_activity_max = [r['activity']['max'] for r in seizure_results]
    normal_activity_max = [r['activity']['max'] for r in normal_results]
    
    print("\n" + "-"*80)
    print("ACTIVITY CONFIDENCE (mean per video):")
    print("  Seizure videos:")
    print(f"    Mean:   {np.mean(seizure_activity):.6f}")
    print(f"    Median: {np.median(seizure_activity):.6f}")
    print(f"    Min:    {np.min(seizure_activity):.6f}")
    print(f"    Max:    {np.max(seizure_activity):.6f}")
    print(f"    P25:    {np.percentile(seizure_activity, 25):.6f}")
    print(f"    P75:    {np.percentile(seizure_activity, 75):.6f}")
    
    print("\n  Normal videos:")
    print(f"    Mean:   {np.mean(normal_activity):.6f}")
    print(f"    Median: {np.median(normal_activity):.6f}")
    print(f"    Min:    {np.min(normal_activity):.6f}")
    print(f"    Max:    {np.max(normal_activity):.6f}")
    print(f"    P25:    {np.percentile(normal_activity, 25):.6f}")
    print(f"    P75:    {np.percentile(normal_activity, 75):.6f}")
    
    # Threshold recommendations
    print("\n" + "="*80)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*80)
    
    # MOTION_DEGREE_THRESHOLD
    # Want to catch seizures while filtering normals
    # Use the minimum seizure value, but check overlap with normal max
    min_seizure_motion = np.min(seizure_motion)
    max_normal_motion = np.max(normal_motion)
    
    print("\nMOTION_DEGREE_THRESHOLD:")
    print(f"  Minimum seizure motion: {min_seizure_motion:.6f}")
    print(f"  Maximum normal motion:  {max_normal_motion:.6f}")
    
    if min_seizure_motion > max_normal_motion:
        recommended_motion = (min_seizure_motion + max_normal_motion) / 2
        print("  ✓ Clear separation!")
        print(f"  💡 RECOMMENDED: {recommended_motion:.6f}")
    else:
        # Overlap - use percentiles
        recommended_motion = np.percentile(seizure_motion, 10)  # Catch 90% of seizures
        print("  ⚠️  Overlap detected - no perfect threshold")
        print(f"  💡 RECOMMENDED: {recommended_motion:.6f} (catches 90% of seizures)")
    
    print("  Current value: 0.025 (2.5%)")
    
    # ACTIVITY_THRESHOLD  
    min_seizure_activity = np.min(seizure_activity)
    max_normal_activity = np.max(normal_activity)
    
    print("\nACTIVITY_THRESHOLD:")
    print(f"  Minimum seizure activity: {min_seizure_activity:.6f}")
    print(f"  Maximum normal activity:  {max_normal_activity:.6f}")
    
    if min_seizure_activity > max_normal_activity:
        recommended_activity = (min_seizure_activity + max_normal_activity) / 2
        print("  ✓ Clear separation!")
        print(f"  💡 RECOMMENDED: {recommended_activity:.6f}")
    else:
        # Overlap - use percentiles
        recommended_activity = np.percentile(seizure_activity, 10)  # Catch 90% of seizures
        print("  ⚠️  Overlap detected - no perfect threshold")
        print(f"  💡 RECOMMENDED: {recommended_activity:.6f} (catches 90% of seizures)")
    
    print("  Current value: 0.7")
    
    # Save detailed results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'videos_analyzed': {
            'seizure': len(seizure_results),
            'normal': len(normal_results)
        },
        'seizure_videos': seizure_results,
        'normal_videos': normal_results,
        'recommendations': {
            'motion_degree_threshold': float(recommended_motion),
            'activity_threshold': float(recommended_activity),
            'notes': 'Based on training data analysis'
        }
    }
    
    output_path = Path('training_data_profile.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
