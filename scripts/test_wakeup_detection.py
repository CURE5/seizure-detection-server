#!/usr/bin/env python3
"""
Test wake-up detection logic on training videos to verify thresholds.

This script:
1. Processes normal and seizure videos
2. Simulates motion magnitude history over time
3. Tests wake-up detection algorithm
4. Reports if wake-ups would be detected correctly
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis import MotionFeatureExtractor

# Wake-up detection thresholds (from server.py)
# Updated Dec 2025 based on video testing
MOTION_SLEEP_THRESHOLD = 0.15   # Below this = sleeping (normal mean=0.16)
MOTION_WAKE_THRESHOLD = 0.50     # Above this = waking up (seizure mean=0.41)
WAKEUP_SLEEP_THRESHOLD_MINUTES = 30  # Minutes of low motion to consider asleep

def test_wakeup_on_video(video_path: Path, fps: float = 5.0):
    """Test wake-up detection on a video."""
    print(f"\n{'='*80}")
    print(f"Testing: {video_path.name}")
    print(f"{'='*80}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return
    
    # Get actual FPS from video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video FPS: {video_fps:.1f}, Total frames: {total_frames}, Duration: {duration_seconds:.1f}s")
    
    # Initialize motion extractor
    extractor = MotionFeatureExtractor()
    
    motion_history = []
    frame_count = 0
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract motion magnitude
        features = extractor.process_frame(frame)
        stats = features.get('stats', {})
        motion_mag = stats.get('avg_magnitude', 0.0)
        motion_history.append(motion_mag)
        
        # Debug first few frames
        if frame_count < 5:
            print(f"  Frame {frame_count}: motion_mag={motion_mag:.4f}, stats={stats}")
        
        prev_frame = frame
        frame_count += 1
        
        # Simulate wake-up detection at processing FPS
        if frame_count % int(video_fps / fps) == 0:
            # Check if we have enough history
            frames_needed = int(WAKEUP_SLEEP_THRESHOLD_MINUTES * 60 * fps)
            
            if len(motion_history) >= frames_needed:
                current_motion = motion_history[-1]
                recent_motion = motion_history[-frames_needed:]
                avg_motion = sum(recent_motion) / len(recent_motion)
                
                # Wake-up detection logic
                if avg_motion < MOTION_SLEEP_THRESHOLD and current_motion > MOTION_WAKE_THRESHOLD:
                    timestamp_sec = frame_count / video_fps
                    print(f"  ⏰ WAKE-UP DETECTED at {timestamp_sec:.1f}s: "
                          f"avg={avg_motion:.3f} < {MOTION_SLEEP_THRESHOLD}, "
                          f"current={current_motion:.3f} > {MOTION_WAKE_THRESHOLD}")
    
    cap.release()
    
    # Summary statistics
    if motion_history:
        motion_array = np.array(motion_history)
        print("\nMotion Statistics:")
        print(f"  Mean: {motion_array.mean():.3f}")
        print(f"  Median: {np.median(motion_array):.3f}")
        print(f"  Min: {motion_array.min():.3f}")
        print(f"  Max: {motion_array.max():.3f}")
        print(f"  Std Dev: {motion_array.std():.3f}")
        
        # Check if video has sleep-like periods
        sleep_frames = np.sum(motion_array < MOTION_SLEEP_THRESHOLD)
        sleep_percentage = (sleep_frames / len(motion_array)) * 100
        wake_frames = np.sum(motion_array > MOTION_WAKE_THRESHOLD)
        wake_percentage = (wake_frames / len(motion_array)) * 100
        
        print("\nThreshold Analysis:")
        print(f"  Sleep-like (<{MOTION_SLEEP_THRESHOLD}): {sleep_frames}/{len(motion_array)} frames ({sleep_percentage:.1f}%)")
        print(f"  Wake-like (>{MOTION_WAKE_THRESHOLD}): {wake_frames}/{len(motion_array)} frames ({wake_percentage:.1f}%)")
        
        # Recommend threshold adjustments if needed
        if motion_array.max() < MOTION_WAKE_THRESHOLD:
            print(f"\n⚠️  WARNING: Max motion {motion_array.max():.3f} never exceeds wake threshold {MOTION_WAKE_THRESHOLD}")
            print(f"  Consider lowering MOTION_WAKE_THRESHOLD to ~{motion_array.max() * 0.8:.2f}")
        
        if motion_array.mean() > MOTION_SLEEP_THRESHOLD:
            print(f"\n⚠️  WARNING: Mean motion {motion_array.mean():.3f} exceeds sleep threshold {MOTION_SLEEP_THRESHOLD}")
            print("  This video may not be suitable for testing sleep detection")


def main():
    """Test wake-up detection on sample videos."""
    training_data_dir = Path(__file__).parent.parent / "training_data"
    
    print("="*80)
    print("WAKE-UP DETECTION THRESHOLD TESTING")
    print("="*80)
    print("\nCurrent Thresholds:")
    print(f"  MOTION_SLEEP_THRESHOLD = {MOTION_SLEEP_THRESHOLD} (below = sleeping)")
    print(f"  MOTION_WAKE_THRESHOLD = {MOTION_WAKE_THRESHOLD} (above = waking)")
    print(f"  WAKEUP_SLEEP_THRESHOLD_MINUTES = {WAKEUP_SLEEP_THRESHOLD_MINUTES} (sleep duration)")
    
    # Test on normal videos (should show mostly low motion)
    normal_dir = training_data_dir / "test" / "normal"
    if normal_dir.exists():
        normal_videos = list(normal_dir.glob("*.mp4"))[:3]  # Test first 3
        print(f"\n\nTesting {len(normal_videos)} NORMAL videos:")
        for video_path in normal_videos:
            test_wakeup_on_video(video_path)
    
    # Test on seizure videos (should show motion spikes)
    seizure_dir = training_data_dir / "test" / "seizure"
    if seizure_dir.exists():
        seizure_videos = list(seizure_dir.glob("*.mp4"))[:3]  # Test first 3
        print(f"\n\nTesting {len(seizure_videos)} SEIZURE videos:")
        for video_path in seizure_videos:
            test_wakeup_on_video(video_path)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. If wake-ups are detected on normal videos: thresholds likely OK")
    print("2. If NO wake-ups detected: lower MOTION_WAKE_THRESHOLD")
    print("3. Check server logs during actual deployment for wake-up events")
    print("4. Consider adjusting WAKEUP_SLEEP_THRESHOLD_MINUTES based on typical sleep duration")


if __name__ == "__main__":
    main()
