#!/usr/bin/env python3
"""
Test the persistence detector on saved seizure sequences.
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))

from seizure_detection.analysis.persistence_detector import PersistenceDetector


def main():
    print("="*80)
    print("PERSISTENCE DETECTOR TEST")
    print("="*80)
    print()
    
    # Load ONNX model
    model_path = Path("models/hypermotor_detector.onnx")
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    
    print(f"✓ Loaded ONNX model: {model_path}\n")
    
    # Initialize persistence detector
    detector = PersistenceDetector(
        window_seconds=30.0,
        confidence_threshold=0.7,
        persistence_threshold=0.80,
        min_duration=15.0
    )
    
    print("Persistence Detector Configuration:")
    print("  Window: 30 seconds")
    print("  Confidence threshold: 0.7")
    print("  Persistence threshold: 80%")
    print("  Min duration: 15 seconds")
    print()
    
    # Test on a seizure video
    seizure_dir = Path("training_data/seizure/processed")
    
    # Get sequences from first seizure video
    seq_files = sorted(seizure_dir.glob("video_20250107_160414_seq*.npy"))
    
    if len(seq_files) == 0:
        print("✗ No sequence files found")
        return
    
    print(f"Testing on seizure video with {len(seq_files)} sequences\n")
    print("="*80)
    
    alert_triggered = False
    alert_time = None
    
    for i, seq_file in enumerate(seq_files):
        # Load sequence
        seq = np.load(seq_file)
        
        # Run ONNX inference
        batch = np.expand_dims(seq, axis=0).astype(np.float32)
        outputs = session.run(None, {input_name: batch})
        
        if len(outputs[0].shape) == 2:
            confidence = float(outputs[0][0, 0])
        else:
            confidence = float(outputs[0][0])
        
        # Simulated timestamp (3 seconds per sequence)
        timestamp = i * 3.0
        
        # Add to detector
        result = detector.add_prediction(confidence, timestamp)
        
        # Print status every 10 sequences
        if i % 10 == 0 or result['seizure_detected']:
            status = "🚨 SEIZURE" if result['seizure_detected'] else "  Normal"
            print(
                f"{status} | T={timestamp:6.1f}s | "
                f"Conf={confidence:.3f} | "
                f"Persist={result['persistence_score']:.1%} | "
                f"Duration={result['duration']:.1f}s | "
                f"Window={result['high_confidence_count']}/{result['window_size']}"
            )
        
        # Track first alert
        if result['seizure_detected'] and not alert_triggered:
            alert_triggered = True
            alert_time = timestamp
            print()
            print("="*80)
            print(f"🚨 FIRST ALERT at {alert_time:.1f}s")
            print(f"   Persistence: {result['persistence_score']:.1%}")
            print(f"   Duration: {result['duration']:.1f}s")
            print(f"   Confidence: {confidence:.3f}")
            print("="*80)
            print()
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    stats = detector.get_stats()
    print(f"Alert triggered: {alert_triggered}")
    if alert_triggered:
        print(f"Time to alert: {alert_time:.1f}s")
    print(f"Final persistence score: {stats['persistence_score']:.1%}")
    print(f"Mean confidence: {stats['mean_confidence']:.3f}")
    print(f"Max confidence: {stats['max_confidence']:.3f}")
    print(f"Seizure duration: {stats['seizure_duration']:.1f}s")
    
    # Test on normal video
    print()
    print("="*80)
    print("Testing on normal video...")
    print("="*80)
    print()
    
    detector.reset()
    
    normal_dir = Path("training_data/normal/processed")
    normal_files = sorted(normal_dir.glob("video_20250107_205040_seq*.npy"))[:40]  # First 40 seqs
    
    if len(normal_files) > 0:
        alert_triggered = False
        
        for i, seq_file in enumerate(normal_files):
            seq = np.load(seq_file)
            batch = np.expand_dims(seq, axis=0).astype(np.float32)
            outputs = session.run(None, {input_name: batch})
            
            if len(outputs[0].shape) == 2:
                confidence = float(outputs[0][0, 0])
            else:
                confidence = float(outputs[0][0])
            
            timestamp = i * 3.0
            result = detector.add_prediction(confidence, timestamp)
            
            if i % 10 == 0 or result['seizure_detected']:
                status = "🚨 SEIZURE" if result['seizure_detected'] else "  Normal"
                print(
                    f"{status} | T={timestamp:6.1f}s | "
                    f"Conf={confidence:.3f} | "
                    f"Persist={result['persistence_score']:.1%} | "
                    f"Duration={result['duration']:.1f}s"
                )
            
            if result['seizure_detected']:
                alert_triggered = True
        
        print()
        stats = detector.get_stats()
        print(f"False alert on normal video: {alert_triggered}")
        print(f"Final persistence score: {stats['persistence_score']:.1%}")
        print(f"Mean confidence: {stats['mean_confidence']:.3f}")


if __name__ == "__main__":
    main()
