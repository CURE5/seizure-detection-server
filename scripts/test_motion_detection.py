#!/usr/bin/env python3
"""
Test motion-based detection pipeline with live visualization.

This script demonstrates the motion feature extraction and visualization
working in real-time on a video file or camera.
"""

import sys
from pathlib import Path
import cv2
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis.motion_features import MotionFeatureExtractor
from seizure_detection.analysis.motion_visualizer import MotionVisualizer
from seizure_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_on_video(video_path: Path, 
                 mode: str = 'all',
                 target_fps: int = 5,
                 save_output: bool = False):
    """
    Test motion detection on a video file.
    
    Args:
        video_path: Path to video file
        mode: Visualization mode ('heatmap', 'vectors', 'mask', 'all')
        target_fps: Target processing FPS
        save_output: Whether to save output video
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {video_path.name}")
    logger.info(f"Resolution: {width}x{height} @ {original_fps:.1f} FPS")
    logger.info(f"Processing at: ~{target_fps} FPS")
    logger.info(f"Visualization mode: {mode}")
    logger.info("\nPress 'q' to quit, SPACE to pause, 'm' to cycle modes")
    
    # Initialize extractors
    extractor = MotionFeatureExtractor(history_length=30)
    visualizer = MotionVisualizer(display_width=width, display_height=height)
    
    # Calculate frame skip
    frame_skip = max(1, int(original_fps / target_fps))
    
    # Setup output video if requested
    output_writer = None
    if save_output:
        output_path = video_path.parent / f"{video_path.stem}_motion_debug.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(
            str(output_path), fourcc, target_fps,
            (width, height + 150)  # Extra space for graph
        )
        logger.info(f"Saving output to: {output_path}")
    
    # Visualization modes
    modes = ['heatmap', 'vectors', 'mask', 'all']
    current_mode_idx = modes.index(mode)
    
    # Processing loop
    frame_count = 0
    processed_count = 0
    motion_history = []
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to match target FPS
            if frame_count % frame_skip != 0:
                continue
            
            # Process frame
            motion_features = extractor.process_frame(frame)
            motion_history.append(motion_features['stats']['avg_magnitude'])
            
            # Keep last 100 values for graph
            if len(motion_history) > 100:
                motion_history = motion_history[-100:]
        else:
            # Use last frame when paused
            pass
        
        # Visualize
        current_mode = modes[current_mode_idx]
        debug_view = visualizer.create_debug_view(
            frame, motion_features, motion_history
        )
        
        # Add mode indicator
        cv2.putText(debug_view, f"Mode: {current_mode.upper()} (press 'm' to change)",
                   (10, debug_view.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Motion Detection Test', debug_view)
        
        # Save if requested
        if output_writer and not paused:
            output_writer.write(debug_view)
        
        processed_count += 1
        
        # Handle keyboard
        key = cv2.waitKey(1 if not paused else 100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            logger.info("Paused" if paused else "Resumed")
        elif key == ord('m'):
            current_mode_idx = (current_mode_idx + 1) % len(modes)
            logger.info(f"Switched to mode: {modes[current_mode_idx]}")
    
    # Cleanup
    cap.release()
    if output_writer:
        output_writer.release()
        logger.info("Output saved!")
    cv2.destroyAllWindows()
    
    logger.info(f"\nProcessed {processed_count} frames from {frame_count} total")


def test_on_camera(camera_id: int = 0,
                  mode: str = 'all'):
    """
    Test motion detection on live camera feed.
    
    Args:
        camera_id: Camera device ID
        mode: Visualization mode
    """
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_id}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Camera {camera_id}: {width}x{height}")
    logger.info(f"Visualization mode: {mode}")
    logger.info("\nPress 'q' to quit, 'm' to cycle modes")
    
    # Initialize
    extractor = MotionFeatureExtractor(history_length=30)
    visualizer = MotionVisualizer(display_width=width, display_height=height)
    
    modes = ['heatmap', 'vectors', 'mask', 'all']
    current_mode_idx = modes.index(mode)
    motion_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process
        motion_features = extractor.process_frame(frame)
        motion_history.append(motion_features['stats']['avg_magnitude'])
        
        if len(motion_history) > 100:
            motion_history = motion_history[-100:]
        
        # Visualize
        current_mode = modes[current_mode_idx]
        debug_view = visualizer.create_debug_view(
            frame, motion_features, motion_history
        )
        
        cv2.putText(debug_view, f"Mode: {current_mode.upper()} (press 'm' to change)",
                   (10, debug_view.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Motion Detection Test', debug_view)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_mode_idx = (current_mode_idx + 1) % len(modes)
            logger.info(f"Switched to mode: {modes[current_mode_idx]}")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test motion-based detection pipeline")
    parser.add_argument("--video", type=Path,
                       help="Path to video file (if not specified, uses camera)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--mode", type=str, default='all',
                       choices=['heatmap', 'vectors', 'mask', 'all'],
                       help="Visualization mode (default: all)")
    parser.add_argument("--fps", type=int, default=5,
                       help="Target processing FPS for video (default: 5)")
    parser.add_argument("--save", action='store_true',
                       help="Save output video with visualizations")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Motion-Based Detection Pipeline Test")
    logger.info("=" * 60)
    
    if args.video:
        if not args.video.exists():
            logger.error(f"Video file not found: {args.video}")
            return 1
        test_on_video(args.video, args.mode, args.fps, args.save)
    else:
        test_on_camera(args.camera, args.mode)
    
    logger.info("\n✅ Test complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
