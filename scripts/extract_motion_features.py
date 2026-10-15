#!/usr/bin/env python3
"""
Extract motion features from video files for training.

This script processes seizure and normal videos to extract motion-based features
suitable for training a seizure detection model.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import json
from typing import Tuple
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis.motion_features import MotionFeatureExtractor
from seizure_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


def extract_motion_from_video(video_path: Path, 
                              output_dir: Path,
                              target_fps: int = 5,
                              sequence_length: int = 30) -> int:
    """
    Extract motion feature sequences from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted sequences
        target_fps: Target frame rate for processing (default: 5)
        sequence_length: Number of frames per sequence (default: 30 = 6 seconds)
        
    Returns:
        Number of sequences extracted
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return 0
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip to achieve target FPS
    frame_skip = max(1, int(original_fps / target_fps))
    
    logger.info(f"Processing {video_path.name}: {original_fps:.1f} FPS, {total_frames} frames")
    logger.info(f"Sampling every {frame_skip} frames to achieve ~{target_fps} FPS")
    
    # Initialize motion extractor
    extractor = MotionFeatureExtractor(history_length=sequence_length)
    
    # Process frames
    frame_count = 0
    sampled_count = 0
    sequences_saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames to match target FPS
        if frame_count % frame_skip != 0:
            continue
        
        # Process frame
        motion_features = extractor.process_frame(frame)
        sampled_count += 1
        
        # Check if we have enough history to save a sequence
        if sampled_count >= sequence_length:
            temporal_features = extractor.get_temporal_features()
            
            if temporal_features is not None:
                # Save sequence
                video_name = video_path.stem
                seq_filename = f"{video_name}_seq{sequences_saved:04d}.npy"
                seq_path = output_dir / seq_filename
                
                np.save(seq_path, temporal_features)
                sequences_saved += 1
                
                # Slide window by 50% (15 frames)
                # This gives us overlapping sequences for more training data
                for _ in range(sequence_length // 2):
                    if len(extractor.motion_magnitude_history) > 0:
                        extractor.motion_magnitude_history.popleft()
                        extractor.motion_density_history.popleft()
                        extractor.motion_direction_history.popleft()
    
    cap.release()
    logger.info(f"Extracted {sequences_saved} sequences from {video_path.name}")
    
    return sequences_saved


def extract_from_directory(video_dir: Path,
                          output_dir: Path,
                          label: str,
                          target_fps: int = 5,
                          sequence_length: int = 30) -> Tuple[int, int]:
    """
    Extract motion features from all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted sequences
        label: Label for these videos ('seizure' or 'normal')
        target_fps: Target frame rate
        sequence_length: Frames per sequence
        
    Returns:
        (number of videos processed, total sequences extracted)
    """
    # Create output directory
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        logger.warning(f"No video files found in {video_dir}")
        return 0, 0
    
    logger.info(f"Found {len(video_files)} videos in {video_dir}")
    
    # Process each video
    total_sequences = 0
    successful_videos = 0
    skipped_videos = 0
    
    for video_path in tqdm(video_files, desc=f"Processing {label} videos"):
        try:
            # Check if this video has already been processed
            video_prefix = f"{video_path.stem}_seq"
            existing_sequences = list(label_dir.glob(f"{video_prefix}*.npy"))
            
            if existing_sequences:
                logger.info(f"Skipping {video_path.name} - already processed ({len(existing_sequences)} sequences found)")
                total_sequences += len(existing_sequences)
                successful_videos += 1
                skipped_videos += 1
                continue
            
            num_sequences = extract_motion_from_video(
                video_path, label_dir, target_fps, sequence_length
            )
            if num_sequences > 0:
                total_sequences += num_sequences
                successful_videos += 1
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            continue
    
    if skipped_videos > 0:
        logger.info(f"Skipped {skipped_videos} already-processed videos")
    logger.info(f"Processed {successful_videos}/{len(video_files)} videos")
    logger.info(f"Total sequences extracted: {total_sequences}")
    
    return successful_videos, total_sequences


def main():
    parser = argparse.ArgumentParser(description="Extract motion features from videos")
    parser.add_argument("--seizure-dir", type=Path, required=True,
                       help="Directory containing seizure videos")
    parser.add_argument("--normal-dir", type=Path, required=True,
                       help="Directory containing normal videos")
    parser.add_argument("--output-dir", type=Path, default=Path("training_data/motion_features"),
                       help="Output directory for extracted features")
    parser.add_argument("--fps", type=int, default=5,
                       help="Target frames per second (default: 5)")
    parser.add_argument("--sequence-length", type=int, default=30,
                       help="Frames per sequence (default: 30 = 6 seconds at 5 FPS)")
    
    args = parser.parse_args()
    
    # Validate input directories
    if not args.seizure_dir.exists():
        logger.error(f"Seizure directory not found: {args.seizure_dir}")
        return 1
    
    if not args.normal_dir.exists():
        logger.error(f"Normal directory not found: {args.normal_dir}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Motion Feature Extraction for Seizure Detection")
    logger.info("=" * 60)
    logger.info(f"Seizure videos: {args.seizure_dir}")
    logger.info(f"Normal videos: {args.normal_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target FPS: {args.fps}")
    logger.info(f"Sequence length: {args.sequence_length} frames ({args.sequence_length/args.fps:.1f} seconds)")
    logger.info("=" * 60)
    
    # Extract seizure videos
    logger.info("\n📹 Processing SEIZURE videos...")
    sz_videos, sz_sequences = extract_from_directory(
        args.seizure_dir, args.output_dir, "seizure",
        args.fps, args.sequence_length
    )
    
    # Extract normal videos
    logger.info("\n📹 Processing NORMAL videos...")
    normal_videos, normal_sequences = extract_from_directory(
        args.normal_dir, args.output_dir, "normal",
        args.fps, args.sequence_length
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Seizure: {sz_videos} videos → {sz_sequences} sequences")
    logger.info(f"Normal: {normal_videos} videos → {normal_sequences} sequences")
    logger.info(f"Total: {sz_videos + normal_videos} videos → {sz_sequences + normal_sequences} sequences")
    
    # Save metadata
    metadata = {
        "extraction_date": str(datetime.now()),
        "fps": args.fps,
        "sequence_length": args.sequence_length,
        "seizure": {
            "videos": sz_videos,
            "sequences": sz_sequences,
            "directory": str(args.seizure_dir)
        },
        "normal": {
            "videos": normal_videos,
            "sequences": normal_sequences,
            "directory": str(args.normal_dir)
        }
    }
    
    metadata_path = args.output_dir / "extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n📝 Metadata saved to: {metadata_path}")
    logger.info("\n✅ Ready for training! Run:")
    logger.info(f"   python scripts/train_motion_model.py --data-dir {args.output_dir}")
    logger.info("\n💡 Note: This script can be safely interrupted (Ctrl+C) and resumed.")
    logger.info("   Already-processed videos will be automatically skipped.")
    
    return 0


if __name__ == "__main__":
    from datetime import datetime
    sys.exit(main())
