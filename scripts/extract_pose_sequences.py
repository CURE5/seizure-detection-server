#!/usr/bin/env python3
"""
Extract pose sequences from sorted training videos.

This script processes videos from training_data/{seizure,normal}/videos/
and extracts pose keypoint sequences for model training.

This is for Hailo primarily, as it only uses 17 keypoints from the full
pose model.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pose detection backend
from seizure_detection.analysis import pose_detection


def load_extraction_progress(progress_file: Path) -> Dict:
    """Load extraction progress from file."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {
        'completed_videos': [],
        'last_updated': None,
        'results': {
            'seizure': [],
            'normal': []
        }
    }


def save_extraction_progress(progress_file: Path, progress: Dict):
    """Save extraction progress to file."""
    progress['last_updated'] = datetime.now().isoformat()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def extract_poses_from_video(
    video_path: Path,
    output_dir: Path,
    max_sequences: Optional[int] = None
) -> Dict:
    """
    Extract pose sequences from a single video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save pose sequences
        max_sequences: Maximum number of sequences to extract (None = all)
    
    Returns:
        dict: Extraction statistics
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {
            'file': video_path.name,
            'error': 'Could not open video',
            'sequences': 0
        }
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_poses = []
    frame_idx = 0
    extracted_frames = 0
    
    print(f"  Processing {video_path.name}...")
    print(f"    FPS: {fps:.1f}, Frames: {frame_count}")
    
    # Setup pose detection
    pose_detection.setup()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with pose detection
            _, pose_data = pose_detection.process_frame(frame)
            
            if pose_data and pose_data.get('pose'):
                # Extract landmarks
                pose_landmarks = pose_data['pose'].get('pose_landmarks', [])
                
                if pose_landmarks:
                    # Get first person's pose (assuming single person)
                    landmarks = pose_landmarks[0] if isinstance(
                        pose_landmarks[0], list
                    ) else pose_landmarks
                    
                    # Convert to numpy array: (17, 3) = keypoints × (x, y, conf)
                    pose_array = np.zeros((17, 3))
                    for i, lm in enumerate(landmarks[:17]):
                        if isinstance(lm, dict):
                            pose_array[i] = [
                                lm.get('x', 0),
                                lm.get('y', 0),
                                lm.get('visibility', 0)
                            ]
                        else:
                            # Handle different landmark format
                            pose_array[i] = [
                                getattr(lm, 'x', 0),
                                getattr(lm, 'y', 0),
                                getattr(lm, 'visibility', 0)
                            ]
                    
                    all_poses.append(pose_array)
                    extracted_frames += 1
            
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 100 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"    Progress: {progress:.1f}% "
                      f"({extracted_frames} poses extracted)", end='\r')
        
        print()  # New line after progress
        
    finally:
        cap.release()
        pose_detection.release()
    
    # Convert to numpy array
    if not all_poses:
        return {
            'file': video_path.name,
            'error': 'No poses detected',
            'sequences': 0,
            'frames_processed': frame_idx,
            'frames_with_pose': 0
        }
    
    all_poses = np.array(all_poses)  # Shape: (T, 17, 3)
    
    # Create sliding window sequences (30 frames = 6 seconds at 5 FPS)
    sequence_length = 30
    stride = 15  # 50% overlap
    sequences = []
    
    for i in range(0, len(all_poses) - sequence_length + 1, stride):
        seq = all_poses[i:i + sequence_length]
        sequences.append(seq)
        
        if max_sequences and len(sequences) >= max_sequences:
            break
    
    # Save sequences
    output_dir.mkdir(parents=True, exist_ok=True)
    video_stem = video_path.stem
    
    saved_sequences = []
    for i, seq in enumerate(sequences):
        output_file = output_dir / f"{video_stem}_seq{i:04d}.npy"
        np.save(output_file, seq)
        saved_sequences.append(str(output_file.name))
    
    return {
        'file': video_path.name,
        'sequences': len(sequences),
        'frames_processed': frame_idx,
        'frames_with_pose': extracted_frames,
        'saved_files': saved_sequences,
        'pose_detection_rate': extracted_frames / frame_idx if frame_idx > 0 else 0
    }


def extract_all_videos(
    seizure_dir: Path,
    normal_dir: Path,
    output_base: Path,
    progress_file: Path
) -> Dict:
    """
    Extract poses from all videos in seizure and normal directories.
    Supports resumption from interruptions.
    
    Returns:
        dict: Summary statistics
    """
    # Load progress
    progress = load_extraction_progress(progress_file)
    completed = set(progress['completed_videos'])
    
    results = {
        'seizure': progress['results']['seizure'],
        'normal': progress['results']['normal'],
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'seizure_videos': 0,
            'normal_videos': 0,
            'seizure_sequences': 0,
            'normal_sequences': 0,
            'errors': []
        }
    }
    
    # Recalculate summary from existing results
    for result in results['seizure']:
        if 'error' in result:
            results['summary']['errors'].append({
                'category': 'seizure',
                'file': result['file'],
                'error': result['error']
            })
        else:
            results['summary']['seizure_sequences'] += result['sequences']
    
    for result in results['normal']:
        if 'error' in result:
            results['summary']['errors'].append({
                'category': 'normal',
                'file': result['file'],
                'error': result['error']
            })
        else:
            results['summary']['normal_sequences'] += result['sequences']
    
    # Process seizure videos
    print("\n" + "="*70)
    print("PROCESSING SEIZURE VIDEOS")
    print("="*70)
    
    seizure_videos = sorted(seizure_dir.glob('*.mp4'))
    seizure_output = output_base / 'seizure' / 'processed'
    
    if completed:
        skipped = sum(1 for v in seizure_videos if v.name in completed)
        print(f"📝 Resuming: {skipped} seizure videos already processed")
    
    for i, video in enumerate(seizure_videos, 1):
        if video.name in completed:
            print(f"[{i}/{len(seizure_videos)}] ⏭️  Skipping {video.name} "
                  f"(already processed)")
            continue
        
        print(f"\n[{i}/{len(seizure_videos)}] Seizure video:")
        
        try:
            result = extract_poses_from_video(video, seizure_output)
            results['seizure'].append(result)
            
            if 'error' in result:
                results['summary']['errors'].append({
                    'category': 'seizure',
                    'file': result['file'],
                    'error': result['error']
                })
            else:
                results['summary']['seizure_sequences'] += result['sequences']
            
            # Mark as completed and save progress
            progress['completed_videos'].append(video.name)
            progress['results']['seizure'] = results['seizure']
            save_extraction_progress(progress_file, progress)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Progress saved.")
            print("   Resume by running the script again.")
            save_extraction_progress(progress_file, progress)
            raise
        except Exception as e:
            print(f"❌ Error processing {video.name}: {e}")
            results['seizure'].append({
                'file': video.name,
                'error': str(e),
                'sequences': 0
            })
            results['summary']['errors'].append({
                'category': 'seizure',
                'file': video.name,
                'error': str(e)
            })
            # Still mark as completed to skip on retry
            progress['completed_videos'].append(video.name)
            progress['results']['seizure'] = results['seizure']
            save_extraction_progress(progress_file, progress)
    
    results['summary']['seizure_videos'] = len(seizure_videos)
    
    # Process normal videos
    print("\n" + "="*70)
    print("PROCESSING NORMAL VIDEOS")
    print("="*70)
    
    normal_videos = sorted(normal_dir.glob('*.mp4'))
    normal_output = output_base / 'normal' / 'processed'
    
    if completed:
        skipped = sum(1 for v in normal_videos if v.name in completed)
        if skipped > 0:
            print(f"📝 Resuming: {skipped} normal videos already processed")
    
    for i, video in enumerate(normal_videos, 1):
        if video.name in completed:
            print(f"[{i}/{len(normal_videos)}] ⏭️  Skipping {video.name} "
                  f"(already processed)")
            continue
        
        print(f"\n[{i}/{len(normal_videos)}] Normal video:")
        
        try:
            result = extract_poses_from_video(video, normal_output)
            results['normal'].append(result)
            
            if 'error' in result:
                results['summary']['errors'].append({
                    'category': 'normal',
                    'file': result['file'],
                    'error': result['error']
                })
            else:
                results['summary']['normal_sequences'] += result['sequences']
            
            # Mark as completed and save progress
            progress['completed_videos'].append(video.name)
            progress['results']['normal'] = results['normal']
            save_extraction_progress(progress_file, progress)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Progress saved.")
            print("   Resume by running the script again.")
            save_extraction_progress(progress_file, progress)
            raise
        except Exception as e:
            print(f"❌ Error processing {video.name}: {e}")
            results['normal'].append({
                'file': video.name,
                'error': str(e),
                'sequences': 0
            })
            results['summary']['errors'].append({
                'category': 'normal',
                'file': video.name,
                'error': str(e)
            })
            # Still mark as completed to skip on retry
            progress['completed_videos'].append(video.name)
            progress['results']['normal'] = results['normal']
            save_extraction_progress(progress_file, progress)
    
    results['summary']['normal_videos'] = len(normal_videos)
    
    return results


def create_dataset_splits(
    seizure_sequences: List[Path],
    normal_sequences: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict:
    """
    Split sequences into train/validation/test sets.
    
    Returns:
        dict: Split assignments
    """
    import random
    
    # Shuffle sequences
    random.seed(42)  # For reproducibility
    seizure_sequences = list(seizure_sequences)
    normal_sequences = list(normal_sequences)
    random.shuffle(seizure_sequences)
    random.shuffle(normal_sequences)
    
    # Calculate split indices
    n_sz = len(seizure_sequences)
    n_norm = len(normal_sequences)
    
    sz_train_end = int(n_sz * train_ratio)
    sz_val_end = sz_train_end + int(n_sz * val_ratio)
    
    norm_train_end = int(n_norm * train_ratio)
    norm_val_end = norm_train_end + int(n_norm * val_ratio)
    
    splits = {
        'train': {
            'seizure': [str(f.name) for f in seizure_sequences[:sz_train_end]],
            'normal': [str(f.name) for f in normal_sequences[:norm_train_end]]
        },
        'validation': {
            'seizure': [str(f.name) for f in seizure_sequences[sz_train_end:sz_val_end]],
            'normal': [str(f.name) for f in normal_sequences[norm_train_end:norm_val_end]]
        },
        'test': {
            'seizure': [str(f.name) for f in seizure_sequences[sz_val_end:]],
            'normal': [str(f.name) for f in normal_sequences[norm_val_end:]]
        }
    }
    
    return splits


def print_summary(results: Dict, splits: Dict):
    """Print extraction and split summary."""
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    
    summary = results['summary']
    
    print("\nVideos Processed:")
    print(f"  Seizure:  {summary['seizure_videos']} videos")
    print(f"  Normal:   {summary['normal_videos']} videos")
    print(f"  Total:    {summary['seizure_videos'] + summary['normal_videos']} videos")
    
    print("\nSequences Extracted:")
    print(f"  Seizure:  {summary['seizure_sequences']} sequences")
    print(f"  Normal:   {summary['normal_sequences']} sequences")
    print(f"  Total:    {summary['seizure_sequences'] + summary['normal_sequences']} sequences")
    
    if summary['errors']:
        print(f"\n⚠️  Errors: {len(summary['errors'])} videos failed")
        for err in summary['errors'][:5]:
            print(f"    - {err['file']}: {err['error']}")
        if len(summary['errors']) > 5:
            print(f"    ... and {len(summary['errors']) - 5} more")
    
    print("\nDataset Splits:")
    for split_name, split_data in splits.items():
        sz_count = len(split_data['seizure'])
        norm_count = len(split_data['normal'])
        total = sz_count + norm_count
        print(f"  {split_name.capitalize()}:")
        print(f"    Seizure: {sz_count}, Normal: {norm_count}, Total: {total}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract pose sequences from training videos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--seizure-dir',
        type=Path,
        default=Path('training_data/seizure/videos'),
        help='Directory containing seizure videos'
    )
    
    parser.add_argument(
        '--normal-dir',
        type=Path,
        default=Path('training_data/normal/videos'),
        help='Directory containing normal videos'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('training_data'),
        help='Base output directory'
    )
    
    parser.add_argument(
        '--results-file',
        type=Path,
        default=Path('training_data/extraction_results.json'),
        help='Save extraction results to JSON file'
    )
    
    parser.add_argument(
        '--splits-file',
        type=Path,
        default=Path('training_data/dataset_splits.json'),
        help='Save dataset splits to JSON file'
    )
    
    parser.add_argument(
        '--progress-file',
        type=Path,
        default=Path('training_data/.extraction_progress.json'),
        help='Progress tracking file'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.seizure_dir.exists():
        print(f"❌ Seizure directory not found: {args.seizure_dir}")
        return 1
    
    if not args.normal_dir.exists():
        print(f"❌ Normal directory not found: {args.normal_dir}")
        return 1
    
    seizure_videos = list(args.seizure_dir.glob('*.mp4'))
    normal_videos = list(args.normal_dir.glob('*.mp4'))
    
    if not seizure_videos:
        print(f"❌ No videos found in {args.seizure_dir}")
        return 1
    
    if not normal_videos:
        print(f"❌ No videos found in {args.normal_dir}")
        return 1
    
    # Check for existing progress
    progress = load_extraction_progress(args.progress_file)
    if progress['completed_videos']:
        print(f"\n📝 Found existing progress: "
              f"{len(progress['completed_videos'])} videos already processed")
        print("   Will resume from where we left off\n")
    
    print(f"Found {len(seizure_videos)} seizure videos")
    print(f"Found {len(normal_videos)} normal videos")
    
    # Extract poses
    try:
        results = extract_all_videos(
            args.seizure_dir,
            args.normal_dir,
            args.output_dir,
            args.progress_file
        )
    except KeyboardInterrupt:
        print("\n\n✅ Progress saved. Run the script again to resume.")
        return 0
    
    # Create dataset splits
    seizure_sequences = sorted(
        (args.output_dir / 'seizure' / 'processed').glob('*.npy')
    )
    normal_sequences = sorted(
        (args.output_dir / 'normal' / 'processed').glob('*.npy')
    )
    
    splits = create_dataset_splits(seizure_sequences, normal_sequences)
    
    # Save results
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(args.splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    # Print summary
    print_summary(results, splits)
    
    print(f"\n💾 Results saved to: {args.results_file}")
    print(f"💾 Dataset splits saved to: {args.splits_file}")
    print("\n" + "="*70)
    print("Next step: python scripts/train_seizure_model.py")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
