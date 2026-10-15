#!/usr/bin/env python3
"""
Test multi-stage detection pipeline.

Demonstrates the 3-stage detection approach:
1. Motion detection (fast screening)
2. Activity analysis (generic pose-based)
3. Custom ONNX model (patient-specific)
"""

import sys
from pathlib import Path
import cv2
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis import MultiStageDetector
from seizure_detection.config import SeizureDetectionConfig


def test_pipeline_with_video(video_path: Path, onnx_model_path: Path):
    """
    Test pipeline on a video file.
    
    Args:
        video_path: Path to test video
        onnx_model_path: Path to ONNX model
    """
    print("="*70)
    print("Multi-Stage Seizure Detection Pipeline Test")
    print("="*70)
    
    # Initialize pipeline
    config = SeizureDetectionConfig()
    detector = MultiStageDetector(
        config=config,
        onnx_model_path=onnx_model_path,
        enable_time_weighting=True
    )
    
    print(f"\n📂 Video: {video_path.name}")
    print(f"📂 Model: {onnx_model_path.name if onnx_model_path.exists() else 'Not found'}")
    print()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("Video info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    print()
    
    # Process frames
    frame_count = 0
    detections = []
    
    print("Processing frames...")
    print("-"*70)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = datetime.now()
        
        # For testing without actual pose detection, create dummy landmarks
        # In production, this would come from Hailo or MediaPipe
        dummy_landmarks = None
        
        # Run pipeline
        result = detector.analyze_frame(frame, dummy_landmarks, timestamp)
        
        # Track detections
        if result.get('alert'):
            detections.append({
                'frame': frame_count,
                'time': timestamp.isoformat(),
                'stage': result['stage_reached'],
                'confidence': result.get('activity', {}).get('confidence', 0.0)
            })
            print(f"🚨 Frame {frame_count}: SEIZURE ALERT!")
        
        # Show progress every 100 frames
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    # Print results
    print("-"*70)
    print("\n✅ Processing complete!")
    print(f"  Frames processed: {frame_count}")
    print(f"  Alerts generated: {len(detections)}")
    print()
    
    # Print statistics
    stats = detector.get_statistics()
    print("Pipeline Statistics:")
    print(f"  Motion triggers: {stats['motion_triggers']}")
    print(f"  Activity triggers: {stats['activity_triggers']}")
    print(f"  Custom model triggers: {stats['custom_triggers']}")
    print(f"  Confirmed seizures: {stats['confirmed_seizures']}")
    print()
    print("Efficiency:")
    print(f"  Motion → Activity: {stats['efficiency']['motion_pass_rate']:.1%}")
    print(f"  Activity → Custom: {stats['efficiency']['activity_pass_rate']:.1%}")
    print()
    print("Configuration:")
    print(f"  Time weighting: {'Enabled' if stats['time_weight_enabled'] else 'Disabled'}")
    print(f"  Custom model: {'Enabled' if stats['custom_model_enabled'] else 'Disabled'}")
    
    # Print detections
    if detections:
        print("\n🚨 Detected Seizures:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. Frame {det['frame']} "
                  f"(Stage {det['stage']}, "
                  f"Confidence: {det['confidence']:.2f})")


def test_pipeline_architecture():
    """Test pipeline architecture without video."""
    print("="*70)
    print("Pipeline Architecture Test")
    print("="*70)
    print()
    
    # Initialize with dummy model path
    config = SeizureDetectionConfig()
    model_path = Path('models/hypermotor_detector.onnx')
    
    detector = MultiStageDetector(
        config=config,
        onnx_model_path=model_path if model_path.exists() else None,
        enable_time_weighting=True
    )
    
    print("Pipeline Stages:")
    for stage in [1, 2, 3]:
        print(f"  Stage {stage}: {detector.get_stage_name(stage)}")
    print()
    
    print("Components:")
    print(f"  Motion Detector: {detector.motion_detector.__class__.__name__}")
    print(f"  Activity Detector: {detector.activity_detector.__class__.__name__}")
    print(f"  Custom Detector: {detector.custom_detector.__class__.__name__ if detector.custom_detector else 'None'}")
    print()
    
    print("Configuration:")
    print(f"  Sequence length: {config.HYPERMOTOR_SEQUENCE_LENGTH} frames")
    print(f"  Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"  Motion threshold: {config.MOTION_DEGREE_THRESHOLD}")
    print()
    
    print("Time-of-Day Weighting:")
    import datetime
    test_times = [
        datetime.time(3, 0),   # 3 AM - normal
        datetime.time(5, 30),  # 5:30 AM - high risk
        datetime.time(7, 0),   # 7 AM - high risk
        datetime.time(12, 0),  # noon - normal
        datetime.time(20, 0),  # 8 PM - normal
    ]
    
    for test_time in test_times:
        # Temporarily set time for testing
        original_method = detector._get_time_weight
        weight = original_method()
        print(f"  {test_time.strftime('%I:%M %p')}: {weight:.2f}x sensitivity")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test multi-stage detection pipeline"
    )
    parser.add_argument(
        '--video',
        type=Path,
        help='Path to test video file'
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('models/hypermotor_detector.onnx'),
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--architecture-only',
        action='store_true',
        help='Test architecture without video'
    )
    
    args = parser.parse_args()
    
    if args.architecture_only or not args.video:
        test_pipeline_architecture()
    else:
        if not args.video.exists():
            print(f"❌ Video not found: {args.video}")
            return 1
        
        test_pipeline_with_video(args.video, args.model)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
