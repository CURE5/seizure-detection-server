#!/usr/bin/env python3
"""
Analyze actual spasm and calm period durations from SAVED training sequences.

Uses the pre-extracted .npy pose sequence files to analyze actual patterns
without needing to re-run pose extraction.
"""

import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis import ONNXSeizureDetector


def analyze_sequence_file(seq_file: Path, model: ONNXSeizureDetector, fps: float = 5.0):
    """Analyze a single .npy sequence file."""
    try:
        # Load sequence
        sequence = np.load(seq_file)
        
        # Run through model
        sequence_input = np.expand_dims(sequence, axis=0).astype(np.float32)
        input_name = model.session.get_inputs()[0].name
        output_name = model.session.get_outputs()[0].name
        
        outputs = model.session.run([output_name], {input_name: sequence_input})
        confidence = float(outputs[0][0][0])
        
        return confidence
    except Exception as e:
        print(f"Error processing {seq_file.name}: {e}")
        return None


def main():
    """Analyze saved training sequences to learn spasm/calm patterns."""
    print("="*80)
    print("SPASM AND CALM PERIOD ANALYSIS (from saved sequences)")
    print("="*80)
    print("\nAnalyzing pre-extracted pose sequences to learn:")
    print("- Real spasm durations")
    print("- Real calm period durations")
    print("- Actual interval patterns\n")
    
    # Initialize model
    model = ONNXSeizureDetector(
        model_path=Path("models/hypermotor_detector.onnx"),
        sequence_length=30,
        confidence_threshold=0.5
    )
    
    if not model.is_ready():
        print("✗ Model not loaded")
        return
    
    # Find saved sequences
    training_data_dir = Path(__file__).parent.parent / "training_data"
    
    # Analyze seizure sequences
    seizure_seqs_dir = training_data_dir / "seizure" / "processed"
    
    if not seizure_seqs_dir.exists():
        print(f"✗ Seizure sequences not found: {seizure_seqs_dir}")
        print("\nNeed to extract sequences first. Run:")
        print("  python scripts/extract_pose_sequences.py")
        return
    
    # Get all sequence files and group by video name
    all_seq_files = sorted(seizure_seqs_dir.glob("video_*.npy"))
    print(f"Found {len(all_seq_files)} total sequence files\n")
    
    # Group sequences by video name (e.g., video_20250107_160414)
    video_groups = {}
    for seq_file in all_seq_files:
        # Extract video name: video_YYYYMMDD_HHMMSS from video_YYYYMMDD_HHMMSS_seqNNNN.npy
        parts = seq_file.stem.split('_')
        if len(parts) >= 4:  # video, date, time, seqNNNN
            video_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(seq_file)
    
    print(f"Grouped into {len(video_groups)} videos\n")
    
    all_video_results = []
    
    for video_name in sorted(video_groups.keys())[:10]:  # Analyze first 10 videos
        seq_files = sorted(video_groups[video_name])
        print(f"\n{'='*80}")
        print(f"Analyzing: {video_name}")
        print(f"{'='*80}")
        
        # Get all sequences for this video (sorted by filename)
        
        if len(seq_files) == 0:
            print("  No sequences found")
            continue
        
        print(f"  Sequences: {len(seq_files)}")
        
        # Run model on each sequence
        predictions = []
        for i, seq_file in enumerate(seq_files):
            conf = analyze_sequence_file(seq_file, model)
            if conf is not None:
                # Each sequence is 30 frames (6s at 5 FPS)
                # Sequences extracted with stride=15, so they overlap by 15 frames (3s)
                # Sequence N starts at frame N*15
                start_frame = i * 15
                start_time = start_frame / 5.0  # 5 FPS
                
                predictions.append({
                    'seq_idx': i,
                    'start_time': start_time,
                    'confidence': conf
                })
        
        if len(predictions) == 0:
            print("  No predictions")
            continue
        
        # Analyze spasm and calm periods
        spasm_threshold = 0.5
        spasm_events = []
        calm_events = []
        
        in_spasm = False
        spasm_start_idx = 0
        spasm_confidences = []
        
        for pred in predictions:
            conf = pred['confidence']
            
            if conf >= spasm_threshold:
                if not in_spasm:
                    # Spasm starting
                    in_spasm = True
                    spasm_start_idx = pred['seq_idx']
                    spasm_confidences = [conf]
                else:
                    # Spasm continuing
                    spasm_confidences.append(conf)
            else:
                if in_spasm:
                    # Spasm ending
                    in_spasm = False
                    
                    # Calculate duration (sequences overlap by 3s, so actual duration is more complex)
                    # Simplification: use number of high-confidence sequences * 3s (stride duration)
                    num_seqs = len(spasm_confidences)
                    duration = num_seqs * 3.0  # 3s per sequence (stride)
                    
                    start_time = predictions[spasm_start_idx]['start_time']
                    end_time = predictions[pred['seq_idx'] - 1]['start_time'] + 6.0  # Add sequence length
                    
                    spasm_events.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'num_sequences': num_seqs,
                        'peak_confidence': max(spasm_confidences),
                        'avg_confidence': np.mean(spasm_confidences)
                    })
                    
                    spasm_confidences = []
        
        # Handle ongoing spasm at end
        if in_spasm and len(spasm_confidences) > 0:
            num_seqs = len(spasm_confidences)
            start_time = predictions[spasm_start_idx]['start_time']
            end_time = predictions[-1]['start_time'] + 6.0
            
            spasm_events.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'num_sequences': num_seqs,
                'peak_confidence': max(spasm_confidences),
                'avg_confidence': np.mean(spasm_confidences)
            })
        
        # Find calm periods
        for i in range(1, len(spasm_events)):
            calm_start = spasm_events[i-1]['end_time']
            calm_end = spasm_events[i]['start_time']
            calm_duration = calm_end - calm_start
            
            if calm_duration > 0:
                calm_events.append({
                    'start_time': calm_start,
                    'end_time': calm_end,
                    'duration': calm_duration
                })
        
        # Display results
        print(f"\n  SPASM EVENTS: {len(spasm_events)}")
        if spasm_events:
            durations = [s['duration'] for s in spasm_events]
            print(f"    Duration: min={min(durations):.1f}s, max={max(durations):.1f}s, mean={np.mean(durations):.1f}s")
            for i, spasm in enumerate(spasm_events, 1):
                print(f"    Spasm {i}: {spasm['start_time']:.1f}s-{spasm['end_time']:.1f}s ({spasm['duration']:.1f}s, {spasm['num_sequences']} seqs)")
        
        print(f"\n  CALM PERIODS: {len(calm_events)}")
        if calm_events:
            durations = [c['duration'] for c in calm_events]
            print(f"    Duration: min={min(durations):.1f}s, max={max(durations):.1f}s, mean={np.mean(durations):.1f}s")
            for i, calm in enumerate(calm_events, 1):
                print(f"    Calm {i}: {calm['start_time']:.1f}s-{calm['end_time']:.1f}s ({calm['duration']:.1f}s)")
        
        all_video_results.append({
            'video': video_name,
            'spasm_events': spasm_events,
            'calm_events': calm_events
        })
    
    # Aggregate statistics
    all_spasm_durations = []
    all_calm_durations = []
    
    for result in all_video_results:
        all_spasm_durations.extend([s['duration'] for s in result['spasm_events']])
        all_calm_durations.extend([c['duration'] for c in result['calm_events']])
    
    print(f"\n\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")
    print(f"\nVideos analyzed: {len(all_video_results)}")
    print(f"Total spasms: {len(all_spasm_durations)}")
    print(f"Total calm periods: {len(all_calm_durations)}")
    
    if all_spasm_durations:
        print("\nSPASM DURATIONS (seconds):")
        print(f"  Min:       {min(all_spasm_durations):.1f}s")
        print(f"  Max:       {max(all_spasm_durations):.1f}s")
        print(f"  Mean:      {np.mean(all_spasm_durations):.1f}s")
        print(f"  Median:    {np.median(all_spasm_durations):.1f}s")
        print(f"  5th %ile:  {np.percentile(all_spasm_durations, 5):.1f}s")
        print(f"  95th %ile: {np.percentile(all_spasm_durations, 95):.1f}s")
    
    if all_calm_durations:
        print("\nCALM PERIOD DURATIONS (seconds):")
        print(f"  Min:       {min(all_calm_durations):.1f}s")
        print(f"  Max:       {max(all_calm_durations):.1f}s")
        print(f"  Mean:      {np.mean(all_calm_durations):.1f}s")
        print(f"  Median:    {np.median(all_calm_durations):.1f}s")
        print(f"  10th %ile: {np.percentile(all_calm_durations, 10):.1f}s")
        print(f"  90th %ile: {np.percentile(all_calm_durations, 90):.1f}s")
    
    # Recommendations
    if all_spasm_durations and all_calm_durations:
        print(f"\n\n{'='*80}")
        print("RECOMMENDED PATTERN DETECTOR SETTINGS")
        print(f"{'='*80}")
        
        spasm_min = np.percentile(all_spasm_durations, 5)
        spasm_max = np.percentile(all_spasm_durations, 95)
        calm_min = np.percentile(all_calm_durations, 10)
        calm_max = np.percentile(all_calm_durations, 90)
        
        print("\nUpdate pattern_detector.py __init__ parameters:")
        print(f"  min_spasm_duration = {spasm_min:.1f}  # seconds")
        print(f"  max_spasm_duration = {spasm_max:.1f}  # seconds")
        print(f"  min_interval = {calm_min:.1f}  # seconds between spasms")
        print(f"  max_interval = {calm_max:.1f}  # seconds between spasms")
    
    # Save results
    output_file = Path(__file__).parent.parent / "spasm_pattern_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'videos_analyzed': len(all_video_results),
            'total_spasms': len(all_spasm_durations),
            'total_calm_periods': len(all_calm_durations),
            'spasm_durations': {
                'data': all_spasm_durations,
                'min': float(min(all_spasm_durations)) if all_spasm_durations else 0,
                'max': float(max(all_spasm_durations)) if all_spasm_durations else 0,
                'mean': float(np.mean(all_spasm_durations)) if all_spasm_durations else 0,
                'median': float(np.median(all_spasm_durations)) if all_spasm_durations else 0,
                'p5': float(np.percentile(all_spasm_durations, 5)) if all_spasm_durations else 0,
                'p95': float(np.percentile(all_spasm_durations, 95)) if all_spasm_durations else 0
            },
            'calm_durations': {
                'data': all_calm_durations,
                'min': float(min(all_calm_durations)) if all_calm_durations else 0,
                'max': float(max(all_calm_durations)) if all_calm_durations else 0,
                'mean': float(np.mean(all_calm_durations)) if all_calm_durations else 0,
                'median': float(np.median(all_calm_durations)) if all_calm_durations else 0,
                'p10': float(np.percentile(all_calm_durations, 10)) if all_calm_durations else 0,
                'p90': float(np.percentile(all_calm_durations, 90)) if all_calm_durations else 0
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
