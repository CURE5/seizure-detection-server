#!/usr/bin/env python3
"""
Test motion model on seizure videos (works without MediaPipe/pose detection).
"""
import sys
import cv2
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis.motion_detector import MotionDetector


def analyze_video_motion(video_path: Path):
    """Analyze video for motion only."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {video_path.name}")
    print(f"{'='*80}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: FPS={fps:.2f}, Frames={total_frames}, Duration={duration:.1f}s\n")
    
    detector = MotionDetector()
    frame_count = 0
    motion_detections = 0
    motion_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect motion
        motion_detected, motion_degree = detector.detect_motion(frame)
        
        if motion_detected:
            motion_detections += 1
        motion_scores.append(motion_degree)
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')
    
    cap.release()
    print()
    
    # Statistics
    motion_scores = np.array(motion_scores)
    print(f"\n{'─'*80}")
    print("MOTION DETECTION RESULTS:")
    print(f"{'─'*80}")
    print(f"Motion detected: {motion_detections} / {frame_count} frames ({motion_detections/frame_count*100:.1f}%)")
    print("\nMotion degree statistics:")
    print(f"  Mean:   {np.mean(motion_scores):.6f}")
    print(f"  Median: {np.median(motion_scores):.6f}")
    print(f"  Max:    {np.max(motion_scores):.6f}")
    print(f"  Min:    {np.min(motion_scores):.6f}")
    print(f"  Std:    {np.std(motion_scores):.6f}")
    print(f"  P75:    {np.percentile(motion_scores, 75):.6f}")
    print(f"  P90:    {np.percentile(motion_scores, 90):.6f}")
    print(f"  P95:    {np.percentile(motion_scores, 95):.6f}")
    print(f"  P99:    {np.percentile(motion_scores, 99):.6f}")
    
    return {
        'video': video_path.name,
        'frames': frame_count,
        'duration': duration,
        'fps': fps,
        'motion_detections': motion_detections,
        'motion_stats': {
            'mean': float(np.mean(motion_scores)),
            'median': float(np.median(motion_scores)),
            'max': float(np.max(motion_scores)),
            'min': float(np.min(motion_scores)),
            'p75': float(np.percentile(motion_scores, 75)),
            'p90': float(np.percentile(motion_scores, 90)),
            'p95': float(np.percentile(motion_scores, 95)),
            'p99': float(np.percentile(motion_scores, 99)),
        }
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, default='training_data/test/seizure')
    parser.add_argument('--output', type=str, default='motion_test_results.json')
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"ERROR: Directory not found: {video_dir}")
        return
    
    video_files = sorted(video_dir.glob('*.mp4'))
    if not video_files:
        print(f"ERROR: No .mp4 files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video(s)")
    
    results = []
    for video_path in video_files:
        result = analyze_video_motion(video_path)
        if result:
            results.append(result)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Summary
    for r in results:
        print(f"{r['video']}:")
        print(f"  Motion detected: {r['motion_detections']}/{r['frames']} frames ({r['motion_detections']/r['frames']*100:.1f}%)")
        print(f"  Peak motion: {r['motion_stats']['max']:.6f}")
        print(f"  P95 motion:  {r['motion_stats']['p95']:.6f}")
        print()


if __name__ == '__main__':
    main()
