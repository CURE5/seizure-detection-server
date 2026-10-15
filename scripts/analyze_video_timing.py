#!/usr/bin/env python3
"""
Analyze recorded videos to check if duration matches FPS and frame count.

This script helps determine if video compression/encoding affects the 
relationship between actual FPS, frame count, and reported duration.
"""

import sys
from pathlib import Path
import cv2
import json
from datetime import timedelta
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_video(video_path: Path) -> Dict:
    """
    Analyze a single video file for timing accuracy.
    
    Returns:
        dict: Video analysis with FPS, frame count, duration, and discrepancies
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {
                'file': video_path.name,
                'error': 'Could not open video'
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        
        # Calculate durations
        reported_duration = frame_count / fps if fps > 0 else 0
        
        # Actually count frames (for verification)
        actual_frame_count = 0
        while True:
            ret = cap.read()[0]
            if not ret:
                break
            actual_frame_count += 1
        
        cap.release()
        
        # Calculate actual duration if frame count differs
        actual_duration = actual_frame_count / fps if fps > 0 else 0
        
        # Calculate discrepancies
        frame_count_match = actual_frame_count == frame_count
        duration_discrepancy = abs(actual_duration - reported_duration)
        
        # Check if duration seems compressed (common issue at low FPS)
        expected_duration_at_5fps = actual_frame_count / 5.0
        compression_ratio = reported_duration / expected_duration_at_5fps if expected_duration_at_5fps > 0 else 0
        
        result = {
            'file': video_path.name,
            'file_size_mb': video_path.stat().st_size / (1024 * 1024),
            'resolution': f"{width}x{height}",
            'codec': codec_str,
            'reported_fps': fps,
            'reported_frame_count': frame_count,
            'actual_frame_count': actual_frame_count,
            'frame_count_match': frame_count_match,
            'reported_duration_sec': reported_duration,
            'actual_duration_sec': actual_duration,
            'duration_discrepancy_sec': duration_discrepancy,
            'reported_duration_str': str(timedelta(seconds=int(reported_duration))),
            'actual_duration_str': str(timedelta(seconds=int(actual_duration))),
            'expected_at_5fps_sec': expected_duration_at_5fps,
            'compression_ratio': compression_ratio,
            'timing_accurate': duration_discrepancy < 1.0,
        }
        
        return result
        
    except Exception as e:
        return {
            'file': video_path.name,
            'error': str(e)
        }


def analyze_directory(video_dir: Path, pattern: str = "*.mp4") -> List[Dict]:
    """
    Analyze all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        pattern: Glob pattern for video files
    
    Returns:
        List of analysis results
    """
    videos = sorted(video_dir.glob(pattern))
    
    if not videos:
        print(f"No videos found in {video_dir} matching pattern '{pattern}'")
        return []
    
    print(f"Found {len(videos)} videos to analyze...")
    print()
    
    results = []
    for i, video_path in enumerate(videos, 1):
        print(f"Analyzing {i}/{len(videos)}: {video_path.name}...", end=" ")
        result = analyze_video(video_path)
        results.append(result)
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
        else:
            status = "✓" if result['timing_accurate'] else "⚠"
            print(f"{status} {result['actual_duration_str']} ({result['actual_frame_count']} frames @ {result['reported_fps']:.1f} FPS)")
    
    return results


def print_summary(results: List[Dict]):
    """Print summary statistics of video analysis."""
    
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    
    if not valid_results:
        print("\n❌ No valid videos to analyze")
        return
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    # Basic stats
    total_videos = len(results)
    valid_videos = len(valid_results)
    error_videos = len(error_results)
    
    print(f"\nTotal videos analyzed: {total_videos}")
    print(f"  ✓ Valid: {valid_videos}")
    print(f"  ❌ Errors: {error_videos}")
    
    if error_videos > 0:
        print("\nVideos with errors:")
        for r in error_results:
            print(f"  - {r['file']}: {r['error']}")
    
    # FPS distribution
    fps_values = [r['reported_fps'] for r in valid_results]
    fps_unique = set(fps_values)
    
    print("\nFPS Distribution:")
    for fps in sorted(fps_unique):
        count = fps_values.count(fps)
        print(f"  {fps:.2f} FPS: {count} videos")
    
    # Duration stats
    durations = [r['actual_duration_sec'] for r in valid_results]
    total_duration = sum(durations)
    avg_duration = total_duration / len(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    
    print("\nDuration Statistics:")
    print(f"  Total recording time: {timedelta(seconds=int(total_duration))}")
    print(f"  Average duration: {timedelta(seconds=int(avg_duration))}")
    print(f"  Shortest: {timedelta(seconds=int(min_duration))}")
    print(f"  Longest: {timedelta(seconds=int(max_duration))}")
    
    # Frame count stats
    total_frames = sum(r['actual_frame_count'] for r in valid_results)
    avg_frames = total_frames / len(valid_results) if valid_results else 0
    
    print("\nFrame Count Statistics:")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Average per video: {avg_frames:.0f} frames")
    
    # Timing accuracy
    accurate = [r for r in valid_results if r['timing_accurate']]
    inaccurate = [r for r in valid_results if not r['timing_accurate']]
    
    print("\nTiming Accuracy:")
    print(f"  ✓ Accurate (<1s discrepancy): {len(accurate)} ({len(accurate)/len(valid_results)*100:.1f}%)")
    print(f"  ⚠ Inaccurate (>1s discrepancy): {len(inaccurate)} ({len(inaccurate)/len(valid_results)*100:.1f}%)")
    
    if inaccurate:
        print("\n  Videos with timing issues:")
        for r in sorted(inaccurate, key=lambda x: x['duration_discrepancy_sec'], reverse=True)[:10]:
            print(f"    - {r['file']}: {r['duration_discrepancy_sec']:.1f}s discrepancy")
    
    # Compression analysis
    compression_ratios = [r['compression_ratio'] for r in valid_results if r['compression_ratio'] > 0]
    if compression_ratios:
        avg_compression = sum(compression_ratios) / len(compression_ratios)
        print("\nCompression Analysis (vs expected 5 FPS):")
        print(f"  Average compression ratio: {avg_compression:.2f}x")
        if avg_compression < 0.9:
            print("  ⚠ Videos appear significantly compressed (expected 1.0x)")
        elif avg_compression > 1.1:
            print("  ⚠ Videos appear expanded/slowed (expected 1.0x)")
        else:
            print("  ✓ Compression ratio looks normal")
    
    # Frame count vs reported
    frame_mismatches = [r for r in valid_results if not r['frame_count_match']]
    if frame_mismatches:
        print("\nFrame Count Mismatches:")
        print(f"  {len(frame_mismatches)} videos have actual != reported frame count")
        for r in frame_mismatches[:5]:
            print(f"    - {r['file']}: reported={r['reported_frame_count']}, actual={r['actual_frame_count']}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze video timing accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all videos in recorded_video directory
  python scripts/analyze_video_timing.py
  
  # Analyze specific directory
  python scripts/analyze_video_timing.py --dir /path/to/videos
  
  # Save results to JSON
  python scripts/analyze_video_timing.py --output analysis.json
  
  # Analyze only .avi files
  python scripts/analyze_video_timing.py --pattern "*.avi"
        """
    )
    
    parser.add_argument(
        '--dir',
        type=Path,
        default=Path('recorded_video'),
        help='Directory containing videos (default: recorded_video/)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mp4',
        help='Glob pattern for video files (default: *.mp4)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information for each video'
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not args.dir.exists():
        print(f"❌ Error: Directory not found: {args.dir}")
        print(f"\nTip: Check if you have recorded videos in {args.dir.absolute()}")
        return 1
    
    # Analyze videos
    results = analyze_directory(args.dir, args.pattern)
    
    if not results:
        return 1
    
    # Print detailed results if verbose
    if args.verbose:
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        for r in results:
            if 'error' not in r:
                print(f"\n{r['file']}:")
                print(f"  Resolution: {r['resolution']}")
                print(f"  Codec: {r['codec']}")
                print(f"  FPS: {r['reported_fps']:.2f}")
                print(f"  Frames: {r['actual_frame_count']} (reported: {r['reported_frame_count']})")
                print(f"  Duration: {r['actual_duration_str']} ({r['actual_duration_sec']:.1f}s)")
                print(f"  Expected @ 5 FPS: {r['expected_at_5fps_sec']:.1f}s")
                print(f"  Compression ratio: {r['compression_ratio']:.2f}x")
                print(f"  Timing accurate: {'✓' if r['timing_accurate'] else '⚠'}")
    
    # Print summary
    print_summary(results)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n" + "="*80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
