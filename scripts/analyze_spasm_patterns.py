#!/usr/bin/env python3
"""
Analyze actual spasm and calm period durations from seizure training videos.

Extracts empirical data about:
- Spasm durations (how long high-confidence periods last)
- Calm periods between spasms (how long low-confidence gaps last)
- Interval patterns (rhythm of spasm→calm→spasm)

This data will inform pattern detector thresholds.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime, timezone, timedelta
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis import ONNXSeizureDetector
from seizure_detection.analysis.pose_detection import process_frame as process_frame_pose


def analyze_spasm_patterns(video_path: Path, pose_threshold: float = 0.5):
    """Analyze spasm and calm periods in a seizure video."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {video_path.name}")
    print(f"{'='*80}")
    
    # Initialize pose model
    model = ONNXSeizureDetector(
        model_path=Path("models/hypermotor_detector.onnx"),
        sequence_length=30,
        confidence_threshold=pose_threshold
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Could not open: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"FPS: {fps}, Frames: {total_frames}, Duration: {duration:.1f}s")
    
    # Collect predictions
    predictions = []
    frame_count = 0
    
    print("Extracting pose predictions...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = datetime.now(timezone.utc) + timedelta(seconds=frame_count / fps)
        
        # Extract pose using process_frame
        try:
            _, pose_telemetry = process_frame_pose(frame, do_draw=False)
            pose_data = pose_telemetry.get("pose", {})
            if pose_data:
                landmarks = pose_data.get("landmarks", [])
                
                if landmarks is not None and len(landmarks) > 0:
                    model.add_pose(landmarks, timestamp)
                    result = model.predict()
                    
                    if result and result.get('ready'):
                        predictions.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'confidence': result['probability']
                        })
        except Exception as e:
            pass  # Skip frames with pose extraction errors
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    if len(predictions) < 30:
        print(f"⚠️  Not enough predictions ({len(predictions)})")
        return None
    
    print(f"✓ Extracted {len(predictions)} predictions")
    
    # Analyze spasm and calm periods
    confidences = np.array([p['confidence'] for p in predictions])
    times = np.array([p['time'] for p in predictions])
    
    # Find spasm events (high confidence periods)
    spasm_events = []
    calm_events = []
    
    in_spasm = False
    spasm_start_idx = 0
    spasm_confidences = []
    
    for i, conf in enumerate(confidences):
        if conf >= pose_threshold:
            if not in_spasm:
                # Spasm starting
                in_spasm = True
                spasm_start_idx = i
                spasm_confidences = [conf]
            else:
                # Spasm continuing
                spasm_confidences.append(conf)
        else:
            if in_spasm:
                # Spasm ending
                in_spasm = False
                
                duration = (times[i-1] - times[spasm_start_idx]) if i > spasm_start_idx else 0
                
                if duration > 0.1:  # At least 100ms
                    spasm_events.append({
                        'start_time': times[spasm_start_idx],
                        'end_time': times[i-1],
                        'duration': duration,
                        'peak_confidence': max(spasm_confidences),
                        'avg_confidence': np.mean(spasm_confidences),
                        'start_idx': spasm_start_idx,
                        'end_idx': i-1
                    })
                
                spasm_confidences = []
    
    # Handle ongoing spasm at end
    if in_spasm and len(spasm_confidences) > 0:
        duration = times[-1] - times[spasm_start_idx]
        if duration > 0.1:
            spasm_events.append({
                'start_time': times[spasm_start_idx],
                'end_time': times[-1],
                'duration': duration,
                'peak_confidence': max(spasm_confidences),
                'avg_confidence': np.mean(spasm_confidences),
                'start_idx': spasm_start_idx,
                'end_idx': len(confidences)-1
            })
    
    # Find calm periods between spasms
    for i in range(1, len(spasm_events)):
        calm_start = spasm_events[i-1]['end_time']
        calm_end = spasm_events[i]['start_time']
        calm_duration = calm_end - calm_start
        
        if calm_duration > 0:
            # Get confidences during calm period
            calm_start_idx = spasm_events[i-1]['end_idx'] + 1
            calm_end_idx = spasm_events[i]['start_idx'] - 1
            
            if calm_end_idx >= calm_start_idx:
                calm_confs = confidences[calm_start_idx:calm_end_idx+1]
                
                calm_events.append({
                    'start_time': calm_start,
                    'end_time': calm_end,
                    'duration': calm_duration,
                    'avg_confidence': np.mean(calm_confs) if len(calm_confs) > 0 else 0.0,
                    'max_confidence': np.max(calm_confs) if len(calm_confs) > 0 else 0.0
                })
    
    # Results
    print(f"\nSPASM EVENTS: {len(spasm_events)}")
    if spasm_events:
        spasm_durations = [s['duration'] for s in spasm_events]
        print(f"  Duration (s): min={min(spasm_durations):.2f}, max={max(spasm_durations):.2f}, "
              f"mean={np.mean(spasm_durations):.2f}, median={np.median(spasm_durations):.2f}")
        print(f"  Confidence: mean={np.mean([s['avg_confidence'] for s in spasm_events]):.3f}")
        
        # Show each spasm
        for i, spasm in enumerate(spasm_events, 1):
            print(f"  Spasm {i}: {spasm['start_time']:.1f}s - {spasm['end_time']:.1f}s "
                  f"({spasm['duration']:.2f}s, conf={spasm['avg_confidence']:.3f})")
    
    print(f"\nCALM PERIODS: {len(calm_events)}")
    if calm_events:
        calm_durations = [c['duration'] for c in calm_events]
        print(f"  Duration (s): min={min(calm_durations):.2f}, max={max(calm_durations):.2f}, "
              f"mean={np.mean(calm_durations):.2f}, median={np.median(calm_durations):.2f}")
        print(f"  Confidence: mean={np.mean([c['avg_confidence'] for c in calm_events]):.3f}")
        
        # Show each calm period
        for i, calm in enumerate(calm_events, 1):
            print(f"  Calm {i}: {calm['start_time']:.1f}s - {calm['end_time']:.1f}s "
                  f"({calm['duration']:.2f}s, avg_conf={calm['avg_confidence']:.3f})")
    
    return {
        'video': video_path.name,
        'duration': duration,
        'spasm_events': spasm_events,
        'calm_events': calm_events,
        'spasm_durations': [s['duration'] for s in spasm_events],
        'calm_durations': [c['duration'] for c in calm_events],
        'num_spasms': len(spasm_events),
        'num_calms': len(calm_events)
    }


def main():
    """Analyze all seizure training videos to learn actual patterns."""
    print("="*80)
    print("SPASM AND CALM PERIOD ANALYSIS")
    print("="*80)
    print("\nAnalyzing actual hypermotor seizure videos to learn:")
    print("- Real spasm durations (not assumptions)")
    print("- Real calm period durations")
    print("- Actual interval patterns\n")
    
    training_data_dir = Path(__file__).parent.parent / "training_data"
    seizure_dir = training_data_dir / "test" / "seizure"
    
    if not seizure_dir.exists():
        print(f"✗ Seizure directory not found: {seizure_dir}")
        return
    
    seizure_videos = list(seizure_dir.glob("*.mp4"))[:10]  # Analyze first 10
    
    if not seizure_videos:
        print("✗ No seizure videos found")
        return
    
    print(f"Analyzing {len(seizure_videos)} seizure videos...\n")
    
    all_results = []
    all_spasm_durations = []
    all_calm_durations = []
    
    for video in seizure_videos:
        result = analyze_spasm_patterns(video)
        if result:
            all_results.append(result)
            all_spasm_durations.extend(result['spasm_durations'])
            all_calm_durations.extend(result['calm_durations'])
    
    # Aggregate statistics
    print(f"\n\n{'='*80}")
    print("AGGREGATE STATISTICS ACROSS ALL VIDEOS")
    print(f"{'='*80}")
    print(f"\nVideos analyzed: {len(all_results)}")
    print(f"Total spasms detected: {sum(r['num_spasms'] for r in all_results)}")
    print(f"Total calm periods: {sum(r['num_calms'] for r in all_results)}")
    
    if all_spasm_durations:
        print("\nSPASM DURATIONS (seconds):")
        print(f"  Min:        {min(all_spasm_durations):.2f}s")
        print(f"  Max:        {max(all_spasm_durations):.2f}s")
        print(f"  Mean:       {np.mean(all_spasm_durations):.2f}s")
        print(f"  Median:     {np.median(all_spasm_durations):.2f}s")
        print(f"  25th %ile:  {np.percentile(all_spasm_durations, 25):.2f}s")
        print(f"  75th %ile:  {np.percentile(all_spasm_durations, 75):.2f}s")
        print(f"  Std Dev:    {np.std(all_spasm_durations):.2f}s")
    
    if all_calm_durations:
        print("\nCALM PERIOD DURATIONS (seconds):")
        print(f"  Min:        {min(all_calm_durations):.2f}s")
        print(f"  Max:        {max(all_calm_durations):.2f}s")
        print(f"  Mean:       {np.mean(all_calm_durations):.2f}s")
        print(f"  Median:     {np.median(all_calm_durations):.2f}s")
        print(f"  25th %ile:  {np.percentile(all_calm_durations, 25):.2f}s")
        print(f"  75th %ile:  {np.percentile(all_calm_durations, 75):.2f}s")
        print(f"  Std Dev:    {np.std(all_calm_durations):.2f}s")
    
    # Recommendations
    print(f"\n\n{'='*80}")
    print("RECOMMENDED PATTERN DETECTOR SETTINGS")
    print(f"{'='*80}")
    
    if all_spasm_durations and all_calm_durations:
        # Use percentiles to set thresholds
        spasm_min = np.percentile(all_spasm_durations, 5)  # 5th percentile
        spasm_max = np.percentile(all_spasm_durations, 95)  # 95th percentile
        calm_min = np.percentile(all_calm_durations, 10)  # 10th percentile
        calm_max = np.percentile(all_calm_durations, 90)  # 90th percentile
        
        print("\nSpasm duration range (captures 90% of spasms):")
        print(f"  min_spasm_duration = {spasm_min:.2f}  # seconds")
        print(f"  max_spasm_duration = {spasm_max:.2f}  # seconds")
        
        print("\nCalm period range (captures 80% of calm periods):")
        print(f"  min_interval = {calm_min:.2f}  # seconds between spasms")
        print(f"  max_interval = {calm_max:.2f}  # seconds between spasms")
        
        print("\nUpdate pattern_detector.py with these values instead of assumptions!")
    
    # Save results to JSON
    output_file = Path(__file__).parent.parent / "spasm_pattern_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'videos_analyzed': len(all_results),
            'total_spasms': sum(r['num_spasms'] for r in all_results),
            'total_calm_periods': sum(r['num_calms'] for r in all_results),
            'spasm_durations': {
                'min': float(min(all_spasm_durations)) if all_spasm_durations else 0,
                'max': float(max(all_spasm_durations)) if all_spasm_durations else 0,
                'mean': float(np.mean(all_spasm_durations)) if all_spasm_durations else 0,
                'median': float(np.median(all_spasm_durations)) if all_spasm_durations else 0,
                'p25': float(np.percentile(all_spasm_durations, 25)) if all_spasm_durations else 0,
                'p75': float(np.percentile(all_spasm_durations, 75)) if all_spasm_durations else 0,
                'std': float(np.std(all_spasm_durations)) if all_spasm_durations else 0,
                'recommended_min': float(np.percentile(all_spasm_durations, 5)) if all_spasm_durations else 0,
                'recommended_max': float(np.percentile(all_spasm_durations, 95)) if all_spasm_durations else 0
            },
            'calm_durations': {
                'min': float(min(all_calm_durations)) if all_calm_durations else 0,
                'max': float(max(all_calm_durations)) if all_calm_durations else 0,
                'mean': float(np.mean(all_calm_durations)) if all_calm_durations else 0,
                'median': float(np.median(all_calm_durations)) if all_calm_durations else 0,
                'p25': float(np.percentile(all_calm_durations, 25)) if all_calm_durations else 0,
                'p75': float(np.percentile(all_calm_durations, 75)) if all_calm_durations else 0,
                'std': float(np.std(all_calm_durations)) if all_calm_durations else 0,
                'recommended_min': float(np.percentile(all_calm_durations, 10)) if all_calm_durations else 0,
                'recommended_max': float(np.percentile(all_calm_durations, 90)) if all_calm_durations else 0
            },
            'details_by_video': all_results
        }, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
