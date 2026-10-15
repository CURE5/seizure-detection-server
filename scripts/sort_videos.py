#!/usr/bin/env python3
"""
Interactive video sorting tool for organizing seizure detection training data.

This script helps you quickly sort videos into seizure/normal categories
by showing video info and letting you make decisions interactively.
"""

import sys
from pathlib import Path
import shutil
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sorted_videos(tracking_file: Path) -> dict:
    """Load previously sorted videos from tracking file."""
    if tracking_file.exists():
        with open(tracking_file) as f:
            return json.load(f)
    return {"seizure": [], "normal": [], "skip": [], "last_processed": 0}


def save_sorted_videos(tracking_file: Path, data: dict):
    """Save sorting progress to tracking file."""
    with open(tracking_file, 'w') as f:
        json.dump(data, f, indent=2)


def get_video_info(video_path: Path) -> dict:
    """Get basic video information without full analysis."""
    stat = video_path.stat()
    return {
        'name': video_path.name,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
    }


def display_video_info(idx: int, total: int, video_info: dict):
    """Display video information in a nice format."""
    print("\n" + "="*70)
    print(f"VIDEO {idx + 1}/{total}")
    print("="*70)
    print(f"Name:     {video_info['name']}")
    print(f"Size:     {video_info['size_mb']:.1f} MB")
    print(f"Modified: {video_info['modified']}")
    print("="*70)


def copy_video(src: Path, category: str, dest_base: Path) -> bool:
    """Copy video to appropriate category folder."""
    if category in ['seizure', 'normal']:
        dest_dir = dest_base / category / 'videos'
        dest_file = dest_dir / src.name
        
        if dest_file.exists():
            print(f"  ⚠️  File already exists in {category}/videos/, skipping copy")
            return True
        
        try:
            shutil.copy2(src, dest_file)
            print(f"  ✓ Copied to {category}/videos/")
            return True
        except Exception as e:
            print(f"  ❌ Error copying: {e}")
            return False
    return True


def interactive_sort(source_dir: Path, dest_dir: Path):
    """
    Interactively sort videos into seizure/normal categories.
    """
    # Get all video files
    videos = sorted(source_dir.glob('*.mp4'))
    
    if not videos:
        print(f"❌ No videos found in {source_dir}")
        return
    
    print(f"\n🎬 Found {len(videos)} videos to sort")
    print("\nCommands:")
    print("  s = Seizure (copy to training_data/seizure/videos/)")
    print("  n = Normal/baseline (copy to training_data/normal/videos/)")
    print("  k = Skip this video")
    print("  q = Quit and save progress")
    print("  b = Go back one video")
    print("  ? = Show this help again")
    
    # Load previous progress
    tracking_file = dest_dir / '.sorting_progress.json'
    progress = load_sorted_videos(tracking_file)
    start_idx = progress['last_processed']
    
    if start_idx > 0:
        print(f"\n📝 Resuming from video {start_idx + 1}")
        print(f"   Previously sorted: {len(progress['seizure'])} seizure, {len(progress['normal'])} normal, {len(progress['skip'])} skipped")
    
    current_idx = start_idx
    
    try:
        while current_idx < len(videos):
            video = videos[current_idx]
            video_info = get_video_info(video)
            
            display_video_info(current_idx, len(videos), video_info)
            
            # Get user input
            choice = input("\nCategory? [s/n/k/q/b/?]: ").lower().strip()
            
            if choice == 'q':
                print("\n💾 Saving progress...")
                progress['last_processed'] = current_idx
                save_sorted_videos(tracking_file, progress)
                print("✓ Progress saved. Resume anytime by running this script again.")
                break
            
            elif choice == 'b':
                if current_idx > 0:
                    current_idx -= 1
                    print("⬅️  Going back...")
                else:
                    print("⚠️  Already at first video")
                continue
            
            elif choice == '?':
                print("\nCommands:")
                print("  s = Seizure")
                print("  n = Normal/baseline")
                print("  k = Skip")
                print("  q = Quit")
                print("  b = Back")
                continue
            
            elif choice in ['s', 'n', 'k']:
                category_name = {
                    's': 'seizure',
                    'n': 'normal',
                    'k': 'skip'
                }[choice]
                
                # Copy file if not skipping
                if category_name != 'skip':
                    success = copy_video(video, category_name, dest_dir)
                    if not success:
                        continue  # Retry same video
                
                # Record decision
                progress[category_name].append(video.name)
                progress['last_processed'] = current_idx + 1
                
                # Save progress every 10 videos
                if (current_idx + 1) % 10 == 0:
                    save_sorted_videos(tracking_file, progress)
                    print("\n💾 Auto-saved progress (every 10 videos)")
                
                current_idx += 1
            
            else:
                print("❌ Invalid choice. Use s/n/k/q/b/?")
        
        # Final save
        save_sorted_videos(tracking_file, progress)
        
        # Summary
        print("\n" + "="*70)
        print("SORTING COMPLETE!")
        print("="*70)
        print(f"Total videos processed: {current_idx}")
        print(f"  ✓ Seizure:  {len(progress['seizure'])} videos")
        print(f"  ✓ Normal:   {len(progress['normal'])} videos")
        print(f"  ⊘ Skipped:  {len(progress['skip'])} videos")
        print("\nVideos copied to:")
        print(f"  {dest_dir}/seizure/videos/")
        print(f"  {dest_dir}/normal/videos/")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving progress...")
        progress['last_processed'] = current_idx
        save_sorted_videos(tracking_file, progress)
        print("✓ Progress saved. Run script again to resume.")
        return


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactively sort videos for training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--source',
        type=Path,
        default=Path('../seizure-detection-app/public/recorded_video'),
        help='Source directory containing videos to sort'
    )
    
    parser.add_argument(
        '--dest',
        type=Path,
        default=Path('training_data'),
        help='Destination base directory (contains seizure/ and normal/ folders)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset sorting progress and start over'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.source.exists():
        print(f"❌ Source directory not found: {args.source}")
        print("   Please check the path or use --source to specify location")
        return 1
    
    if not args.dest.exists():
        print(f"❌ Destination directory not found: {args.dest}")
        print(f"   Run this first: mkdir -p {args.dest}/{{seizure,normal}}/videos")
        return 1
    
    # Check if destination folders exist
    for category in ['seizure', 'normal']:
        folder = args.dest / category / 'videos'
        if not folder.exists():
            print(f"❌ Missing folder: {folder}")
            print(f"   Run: mkdir -p {folder}")
            return 1
    
    # Reset progress if requested
    if args.reset:
        tracking_file = args.dest / '.sorting_progress.json'
        if tracking_file.exists():
            tracking_file.unlink()
            print("✓ Progress reset")
    
    # Start interactive sorting
    interactive_sort(args.source, args.dest)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
