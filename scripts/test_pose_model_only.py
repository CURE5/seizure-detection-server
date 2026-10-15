#!/usr/bin/env python3
"""
Test ONLY the pose model directly on videos, bypassing all gates.
This tests if pose detection + model inference is working at all.
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis.pose_detection import setup as setup_pose, release as release_pose, process_frame
from seizure_detection.analysis.onnx_detector import ONNXSeizureDetector
from seizure_detection.config import SeizureDetectionConfig


def test_video(video_path: Path):
    """Test pose model on a single video."""
    print(f"\n{'='*80}")
    print(f"Testing: {video_path.name}")
    print(f"{'='*80}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {total_frames} frames @ {fps:.1f}fps = {duration:.1f}s")
    
    # Initialize pose model
    config = SeizureDetectionConfig()
    detector = ONNXSeizureDetector(
        model_path=Path('models/hypermotor_detector.onnx'),
        sequence_length=config.HYPERMOTOR_SEQUENCE_LENGTH,
        confidence_threshold=0.5
    )
    
    poses_extracted = 0
    poses_with_person = 0
    model_predictions = 0
    seizure_detections = 0
    confidences = []
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        timestamp = datetime.fromtimestamp(frame_num / fps if fps > 0 else frame_num)
        
        # Extract pose
        try:
            _, telemetry = process_frame(frame, do_draw=False)
            pose_data = telemetry.get("pose", {})
            
            if pose_data:
                poses_extracted += 1
                landmarks = pose_data.get("landmarks", [])
                
                if landmarks and len(landmarks) > 0:
                    poses_with_person += 1
                    
                    # Add to model
                    detector.add_pose(landmarks, timestamp)
                    
                    # Try prediction
                    if detector.is_ready():
                        result = detector.predict()
                        if result and result.get('ready'):
                            model_predictions += 1
                            conf = result['probability']
                            confidences.append(conf)
                            
                            if conf >= 0.5:
                                seizure_detections += 1
                                print(f"  Frame {frame_num}: SEIZURE DETECTED (conf={conf:.4f})")
        except Exception as e:
            print(f"  Frame {frame_num}: Error - {e}")
        
        if frame_num % 50 == 0:
            print(f"  Progress: {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)", end='\r')
    
    cap.release()
    print()
    
    # Results
    print(f"\n{'─'*80}")
    print("RESULTS:")
    print(f"{'─'*80}")
    print(f"Frames processed:          {frame_num}")
    print(f"Poses extracted:           {poses_extracted} ({poses_extracted/frame_num*100:.1f}%)")
    print(f"Poses with person:         {poses_with_person} ({poses_with_person/frame_num*100:.1f}%)")
    print(f"Model predictions:         {model_predictions}")
    print(f"Seizure detections:        {seizure_detections} ({seizure_detections/frame_num*100:.1f}%)")
    
    if confidences:
        print(f"\n{'─'*80}")
        print("CONFIDENCE SCORES:")
        print(f"{'─'*80}")
        print(f"  Mean:   {np.mean(confidences):.4f}")
        print(f"  Median: {np.median(confidences):.4f}")
        print(f"  Max:    {np.max(confidences):.4f}")
        print(f"  Min:    {np.min(confidences):.4f}")
        print(f"  P95:    {np.percentile(confidences, 95):.4f}")
        print(f"  P99:    {np.percentile(confidences, 99):.4f}")
        
        # Analyze threshold
        print(f"\n{'─'*80}")
        print("THRESHOLD ANALYSIS:")
        print(f"{'─'*80}")
        above_05 = sum(1 for c in confidences if c >= 0.5)
        above_03 = sum(1 for c in confidences if c >= 0.3)
        above_07 = sum(1 for c in confidences if c >= 0.7)
        print(f"  >= 0.3: {above_03} predictions ({above_03/len(confidences)*100:.1f}%)")
        print(f"  >= 0.5: {above_05} predictions ({above_05/len(confidences)*100:.1f}%)")
        print(f"  >= 0.7: {above_07} predictions ({above_07/len(confidences)*100:.1f}%)")
    else:
        print("\n⚠️  NO MODEL PREDICTIONS - Model never had enough frames")


def main():
    """Main function."""
    print("Initializing pose detection...")
    setup_pose()
    
    video_dir = Path('training_data/test/seizure')
    videos = sorted(video_dir.glob('*.mp4'))
    
    print(f"\nFound {len(videos)} video(s)")
    
    for video_path in videos:
        test_video(video_path)
    
    release_pose()
    print("\n✓ Done")


if __name__ == '__main__':
    main()
