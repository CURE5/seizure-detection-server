#!/usr/bin/env python3
"""
Analyze statistical differences between seizure and normal sequences.

Instead of assuming rhythmic patterns, find what features actually
distinguish seizure activity from normal behavior.

Features to analyze:
- Confidence distribution (mean, variance, percentiles)
- Temporal patterns (autocorrelation, transitions)
- Sequence-to-sequence changes (derivatives, volatility)
- Persistence (how long high/low confidence lasts)
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
import json
import onnxruntime as ort

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_features(predictions):
    """Extract statistical features from prediction sequence."""
    confidences = np.array([p['confidence'] for p in predictions])
    
    if len(confidences) == 0:
        return None
    
    # Basic statistics
    features = {
        'mean': float(np.mean(confidences)),
        'std': float(np.std(confidences)),
        'min': float(np.min(confidences)),
        'max': float(np.max(confidences)),
        'median': float(np.median(confidences)),
        'q25': float(np.percentile(confidences, 25)),
        'q75': float(np.percentile(confidences, 75)),
        'iqr': float(np.percentile(confidences, 75) - np.percentile(confidences, 25)),
    }
    
    # Temporal features (if enough data)
    if len(confidences) >= 3:
        # Changes between consecutive predictions
        diffs = np.diff(confidences)
        features['mean_change'] = float(np.mean(np.abs(diffs)))
        features['max_change'] = float(np.max(np.abs(diffs)))
        features['volatility'] = float(np.std(diffs))
        
        # Persistence (autocorrelation at lag 1)
        if len(confidences) >= 4:
            autocorr = np.corrcoef(confidences[:-1], confidences[1:])[0, 1]
            features['autocorr'] = float(autocorr) if not np.isnan(autocorr) else 0.0
        
        # Time above/below thresholds
        features['frac_above_05'] = float(np.mean(confidences > 0.5))
        features['frac_above_07'] = float(np.mean(confidences > 0.7))
        features['frac_above_09'] = float(np.mean(confidences > 0.9))
        features['frac_below_03'] = float(np.mean(confidences < 0.3))
        
        # Longest runs
        above_05 = confidences > 0.5
        if np.any(above_05):
            # Find longest continuous run above 0.5
            runs = []
            current_run = 0
            for val in above_05:
                if val:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            features['max_run_above_05'] = int(max(runs)) if runs else 0
        else:
            features['max_run_above_05'] = 0
    
    return features


def analyze_video(seq_files, session, input_name, label):
    """Analyze all sequences from one video."""
    predictions = []
    
    for i, seq_file in enumerate(sorted(seq_files)):
        try:
            seq = np.load(seq_file)
            # Shape should be (30, 33, 3)
            if seq.shape != (30, 33, 3):
                print(f"  Unexpected shape {seq.shape} for {seq_file.name}")
                continue
            
            # Add batch dimension and run inference
            batch = np.expand_dims(seq, axis=0).astype(np.float32)
            outputs = session.run(None, {input_name: batch})
            
            # Get confidence (first element of output)
            if len(outputs[0].shape) == 2:
                conf = float(outputs[0][0, 0])
            else:
                conf = float(outputs[0][0])
            
            # Sequences extracted with stride=15 (3s actual time between starts)
            start_time = i * 3.0
            predictions.append({
                'seq_idx': i,
                'start_time': start_time,
                'confidence': conf
            })
        except Exception as e:
            print(f"  Error loading {seq_file.name}: {e}")
            continue
    
    if len(predictions) == 0:
        return None
    
    features = extract_features(predictions)
    if features:
        features['label'] = label
        features['num_sequences'] = len(predictions)
        features['duration_seconds'] = len(predictions) * 3.0
    
    return features


def main():
    print("="*80)
    print("FEATURE ANALYSIS: Seizure vs Normal")
    print("="*80)
    print("\nFind statistical differences without assumptions")
    print("- What makes seizure sequences different from normal?")
    print("- What features are most discriminative?")
    print()
    
    # Initialize model
    model_path = Path("models/hypermotor_detector.onnx")
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return
    
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    print(f"✓ Loaded ONNX model: {model_path}")
    print(f"  Input: {input_name}, Shape: {session.get_inputs()[0].shape}\n")
    
    training_data_dir = Path(__file__).parent.parent / "training_data"
    
    # Load seizure sequences
    seizure_dir = training_data_dir / "seizure" / "processed"
    normal_dir = training_data_dir / "normal" / "processed"
    
    if not seizure_dir.exists():
        print(f"✗ Seizure sequences not found: {seizure_dir}")
        return
    
    if not normal_dir.exists():
        print(f"✗ Normal sequences not found: {normal_dir}")
        return
    
    # Group seizure sequences by video
    seizure_files = sorted(seizure_dir.glob("video_*.npy"))
    seizure_groups = {}
    for seq_file in seizure_files:
        parts = seq_file.stem.split('_')
        if len(parts) >= 4:
            video_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if video_name not in seizure_groups:
                seizure_groups[video_name] = []
            seizure_groups[video_name].append(seq_file)
    
    # Group normal sequences by video
    normal_files = sorted(normal_dir.glob("video_*.npy"))
    normal_groups = {}
    for seq_file in normal_files:
        parts = seq_file.stem.split('_')
        if len(parts) >= 4:
            video_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if video_name not in normal_groups:
                normal_groups[video_name] = []
            normal_groups[video_name].append(seq_file)
    
    print(f"Found {len(seizure_groups)} seizure videos")
    print(f"Found {len(normal_groups)} normal videos\n")
    
    # Analyze seizure videos
    print("Analyzing seizure videos...")
    seizure_features = []
    for video_name in sorted(seizure_groups.keys())[:20]:  # First 20
        seq_files = seizure_groups[video_name]
        features = analyze_video(seq_files, session, input_name, 'seizure')
        if features:
            seizure_features.append(features)
            print(f"  ✓ {video_name}: {features['num_sequences']} seqs, "
                  f"mean conf={features['mean']:.3f}, std={features['std']:.3f}")
    
    print("\nAnalyzing normal videos...")
    normal_features = []
    for video_name in sorted(normal_groups.keys())[:20]:  # First 20
        seq_files = normal_groups[video_name]
        features = analyze_video(seq_files, session, input_name, 'normal')
        if features:
            normal_features.append(features)
            print(f"  ✓ {video_name}: {features['num_sequences']} seqs, "
                  f"mean conf={features['mean']:.3f}, std={features['std']:.3f}")
    
    # Compare distributions
    print(f"\n{'='*80}")
    print("STATISTICAL COMPARISON")
    print(f"{'='*80}\n")
    
    feature_names = [
        'mean', 'std', 'min', 'max', 'median', 'iqr',
        'mean_change', 'max_change', 'volatility', 'autocorr',
        'frac_above_05', 'frac_above_07', 'frac_above_09', 'frac_below_03',
        'max_run_above_05'
    ]
    
    results = {
        'seizure_stats': {},
        'normal_stats': {},
        'differences': {},
        'discriminative_power': {}
    }
    
    for feature in feature_names:
        # Extract values
        seizure_vals = [f[feature] for f in seizure_features if feature in f]
        normal_vals = [f[feature] for f in normal_features if feature in f]
        
        if len(seizure_vals) == 0 or len(normal_vals) == 0:
            continue
        
        # Statistics
        sz_mean = np.mean(seizure_vals)
        sz_std = np.std(seizure_vals)
        norm_mean = np.mean(normal_vals)
        norm_std = np.std(normal_vals)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((sz_std**2 + norm_std**2) / 2)
        cohens_d = (sz_mean - norm_mean) / pooled_std if pooled_std > 0 else 0
        
        # Statistical test (t-test)
        t_stat, p_value = stats.ttest_ind(seizure_vals, normal_vals)
        
        results['seizure_stats'][feature] = {
            'mean': float(sz_mean),
            'std': float(sz_std),
            'min': float(np.min(seizure_vals)),
            'max': float(np.max(seizure_vals))
        }
        
        results['normal_stats'][feature] = {
            'mean': float(norm_mean),
            'std': float(norm_std),
            'min': float(np.min(normal_vals)),
            'max': float(np.max(normal_vals))
        }
        
        results['differences'][feature] = {
            'cohens_d': float(cohens_d),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
        
        results['discriminative_power'][feature] = float(abs(cohens_d))
        
        # Print results
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{feature:20s} | Seizure: {sz_mean:7.3f}±{sz_std:.3f} | "
              f"Normal: {norm_mean:7.3f}±{norm_std:.3f} | "
              f"Cohen's d: {cohens_d:6.2f} {sig_marker}")
    
    # Rank features by discriminative power
    print(f"\n{'='*80}")
    print("MOST DISCRIMINATIVE FEATURES (by effect size)")
    print(f"{'='*80}\n")
    
    ranked = sorted(results['discriminative_power'].items(), 
                   key=lambda x: x[1], reverse=True)
    
    for i, (feature, effect_size) in enumerate(ranked[:10], 1):
        diff = results['differences'][feature]
        sz = results['seizure_stats'][feature]
        norm = results['normal_stats'][feature]
        
        print(f"{i}. {feature}")
        print(f"   Effect size: {effect_size:.2f} (Cohen's d)")
        print(f"   Seizure:     {sz['mean']:.3f} ± {sz['std']:.3f}")
        print(f"   Normal:      {norm['mean']:.3f} ± {norm['std']:.3f}")
        print(f"   p-value:     {diff['p_value']:.6f}")
        print()
    
    # Save results
    output_file = Path("feature_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_file.absolute()}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    top_features = ranked[:3]
    print("Top 3 most discriminative features:")
    for i, (feature, effect_size) in enumerate(top_features, 1):
        print(f"  {i}. {feature} (effect size: {effect_size:.2f})")
    
    print("\nPattern detection strategy:")
    print("  Instead of assuming rhythmic intervals (4-20s),")
    print("  use these statistical features to detect anomalies:")
    print()
    for feature, _ in top_features:
        sz_mean = results['seizure_stats'][feature]['mean']
        norm_mean = results['normal_stats'][feature]['mean']
        if sz_mean > norm_mean:
            print(f"  - Alert when {feature} > threshold (seizure typically higher)")
        else:
            print(f"  - Alert when {feature} < threshold (seizure typically lower)")


if __name__ == "__main__":
    main()
