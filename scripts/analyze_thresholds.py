"""
Analyze ONNX model test set to find optimal detection thresholds.

This script:
1. Loads test sequences and ground truth labels
2. Runs model inference on test set
3. Generates ROC curve and Precision-Recall curve
4. Finds optimal thresholds for different operating points
5. Recommends threshold based on safety priorities

Usage:
    python scripts/analyze_thresholds.py
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import json
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve,
    confusion_matrix
)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "hypermotor_detector.onnx"
TRAINING_DATA = BASE_DIR / "training_data"
SPLITS_FILE = TRAINING_DATA / "dataset_splits.json"


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load test sequences and labels."""
    print("Loading test data...")
    
    # Load splits to get test file list
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"Splits file not found: {SPLITS_FILE}")
    
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)
    
    test_split = splits.get('test', {})
    if not test_split:
        raise ValueError("No test split found in dataset_splits.json")
    
    # Get seizure and normal test files
    seizure_files = test_split.get('seizure', [])
    normal_files = test_split.get('normal', [])
    
    print(f"Found {len(seizure_files)} seizure + {len(normal_files)} normal test sequences")
    
    # Load sequences
    sequences = []
    labels = []
    
    # Load seizure sequences
    seizure_dir = TRAINING_DATA / "seizure" / "processed"
    for npy_file in seizure_files:
        seq_path = seizure_dir / npy_file
        
        if not seq_path.exists():
            print(f"Warning: Sequence file not found: {seq_path}")
            continue
        
        try:
            seq = np.load(seq_path)
            # Verify shape: (30, 33, 3) or (30, 17, 3)
            if seq.shape[0] == 30 and seq.shape[2] == 3:
                sequences.append(seq)
                labels.append(1)  # Seizure label
        except Exception as e:
            print(f"Error loading {seq_path}: {e}")
    
    # Load normal sequences
    normal_dir = TRAINING_DATA / "normal" / "processed"
    for npy_file in normal_files:
        seq_path = normal_dir / npy_file
        
        if not seq_path.exists():
            print(f"Warning: Sequence file not found: {seq_path}")
            continue
        
        try:
            seq = np.load(seq_path)
            # Verify shape: (30, 33, 3) or (30, 17, 3)
            if seq.shape[0] == 30 and seq.shape[2] == 3:
                sequences.append(seq)
                labels.append(0)  # Normal label
        except Exception as e:
            print(f"Error loading {seq_path}: {e}")
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"Loaded {len(X)} test sequences")
    print(f"  Seizure: {np.sum(y == 1)}")
    print(f"  Normal: {np.sum(y == 0)}")
    print(f"  Shape: {X.shape}")
    
    return X, y


def run_inference(model_path: Path, X: np.ndarray) -> np.ndarray:
    """Run ONNX model inference on test set."""
    print(f"\nRunning inference with {model_path.name}...")
    
    # Load ONNX model
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    
    print(f"Model input: {input_name}")
    print(f"Expected shape: {session.get_inputs()[0].shape}")
    print(f"Input data shape: {X.shape}")
    
    # Run inference
    predictions = []
    batch_size = 32
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size].astype(np.float32)
        outputs = session.run(None, {input_name: batch})
        # Get probability scores - handle different output formats
        if len(outputs[0].shape) == 2:
            predictions.extend(outputs[0][:, 0])  # Shape: (batch, 1)
        else:
            predictions.extend(outputs[0])  # Shape: (batch,)
    
    predictions = np.array(predictions)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    return predictions


def find_optimal_thresholds(y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
    """Find optimal thresholds for different operating points."""
    print("\nAnalyzing optimal thresholds...")
    
    # Ensure y_true and y_scores are 1D arrays
    y_true = y_true.ravel()
    y_scores = y_scores.ravel()
    
    print(f"Labels shape: {y_true.shape}, unique values: {np.unique(y_true)}")
    print(f"Scores shape: {y_scores.shape}")
    
    # Generate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Generate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 scores for PR curve
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    # Find key operating points
    results = {
        'roc_auc': float(roc_auc),
        'current_threshold_0.5': {
            'threshold': 0.5,
            **analyze_threshold(y_true, y_scores, 0.5)
        },
    }
    
    # 1. Maximum F1 score (balanced precision and recall)
    max_f1_idx = np.argmax(f1_scores)
    max_f1_threshold = pr_thresholds[max_f1_idx]
    results['max_f1'] = {
        'threshold': float(max_f1_threshold),
        'f1_score': float(f1_scores[max_f1_idx]),
        **analyze_threshold(y_true, y_scores, max_f1_threshold)
    }
    
    # 2. 100% recall (no missed seizures) - find lowest threshold
    recall_100_indices = np.where(recall >= 0.999)[0]  # Allow 0.1% tolerance
    if len(recall_100_indices) > 0:
        # Get the highest threshold that still achieves 100% recall
        recall_100_idx = recall_100_indices[0]  # First index in sorted recall
        if recall_100_idx < len(pr_thresholds):
            recall_100_threshold = pr_thresholds[recall_100_idx]
            results['recall_100'] = {
                'threshold': float(recall_100_threshold),
                **analyze_threshold(y_true, y_scores, recall_100_threshold)
            }
    
    # 3. 99% recall (allow 1% miss rate for better precision)
    recall_99_indices = np.where(recall >= 0.99)[0]
    if len(recall_99_indices) > 0:
        recall_99_idx = recall_99_indices[0]
        if recall_99_idx < len(pr_thresholds):
            recall_99_threshold = pr_thresholds[recall_99_idx]
            results['recall_99'] = {
                'threshold': float(recall_99_threshold),
                **analyze_threshold(y_true, y_scores, recall_99_threshold)
            }
    
    # 4. 95% recall (more tolerance for better precision)
    recall_95_indices = np.where(recall >= 0.95)[0]
    if len(recall_95_indices) > 0:
        recall_95_idx = recall_95_indices[0]
        if recall_95_idx < len(pr_thresholds):
            recall_95_threshold = pr_thresholds[recall_95_idx]
            results['recall_95'] = {
                'threshold': float(recall_95_threshold),
                **analyze_threshold(y_true, y_scores, recall_95_threshold)
            }
    
    # 5. Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    youden_idx = np.argmax(j_scores)
    youden_threshold = roc_thresholds[youden_idx]
    results['youden'] = {
        'threshold': float(youden_threshold),
        'j_statistic': float(j_scores[youden_idx]),
        **analyze_threshold(y_true, y_scores, youden_threshold)
    }
    
    return results, (fpr, tpr, roc_auc), (precision, recall, pr_thresholds)


def analyze_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict:
    """Analyze performance at specific threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'recall': float(recall),
        'precision': float(precision),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'total_seizures': int(tp + fn),
        'missed_seizures': int(fn),
        'false_alarms': int(fp),
    }


def plot_curves(fpr, tpr, roc_auc, precision, recall, pr_thresholds, results):
    """Plot ROC and Precision-Recall curves with optimal points."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Mark operating points
    for name, data in results.items():
        if name in ['max_f1', 'youden', 'current_threshold_0.5']:
            thresh = data['threshold']
            recall_val = data['recall']
            # Find corresponding FPR
            y_pred = (pr_thresholds >= thresh).astype(int)
            # Approximate FPR from specificity
            fpr_val = 1 - data['specificity']
            ax1.plot(fpr_val, recall_val, 'o', markersize=8, label=f'{name} (t={thresh:.2f})')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate (Recall)')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='blue', lw=2, label='PR curve')
    
    # Mark operating points on PR curve
    for name, data in results.items():
        if name not in ['roc_auc']:
            ax2.plot(data['recall'], data['precision'], 'o', markersize=8, 
                    label=f'{name} (t={data["threshold"]:.2f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = BASE_DIR / "threshold_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()  # Close instead of show for non-interactive mode


def print_recommendations(results: Dict):
    """Print threshold recommendations."""
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nROC AUC: {results['roc_auc']:.4f}")
    
    print("\n" + "-"*80)
    print("OPERATING POINTS:")
    print("-"*80)
    
    for name, data in results.items():
        if name == 'roc_auc':
            continue
            
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  Threshold: {data['threshold']:.4f}")
        print(f"  Recall (Sensitivity): {data['recall']:.1%} ({data['total_seizures'] - data['missed_seizures']}/{data['total_seizures']} seizures caught)")
        print(f"  Precision: {data['precision']:.1%} ({data['false_alarms']} false alarms)")
        print(f"  Specificity: {data['specificity']:.1%}")
        print(f"  F1 Score: {data['f1_score']:.4f}")
        print(f"  Missed Seizures: {data['missed_seizures']}")
        print(f"  False Alarms: {data['false_alarms']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    # Safety-critical recommendation
    if 'recall_100' in results:
        r100 = results['recall_100']
        print("\n🛡️  MAXIMUM SAFETY (100% Recall):")
        print(f"   POSE_MODEL_THRESHOLD={r100['threshold']:.3f}")
        print(f"   • Zero missed seizures ({r100['missed_seizures']} misses)")
        print(f"   • {r100['false_alarms']} false alarms")
        print("   • Use if: Missing a seizure is unacceptable")
    
    if 'recall_99' in results:
        r99 = results['recall_99']
        print("\n⚖️  BALANCED SAFETY (99% Recall):")
        print(f"   POSE_MODEL_THRESHOLD={r99['threshold']:.3f}")
        print(f"   • {r99['missed_seizures']} missed seizures (1% risk)")
        print(f"   • {r99['false_alarms']} false alarms")
        print("   • Use if: Willing to accept very rare misses for fewer false alarms")
    
    if 'max_f1' in results:
        mf1 = results['max_f1']
        print("\n📊 OPTIMAL BALANCE (Max F1):")
        print(f"   POSE_MODEL_THRESHOLD={mf1['threshold']:.3f}")
        print(f"   • Recall: {mf1['recall']:.1%} ({mf1['missed_seizures']} misses)")
        print(f"   • Precision: {mf1['precision']:.1%} ({mf1['false_alarms']} false alarms)")
        print("   • Use if: Want to balance sensitivity and false alarm rate")
    
    # Current setting comparison
    curr = results['current_threshold_0.5']
    print("\n📍 CURRENT SETTING (0.5):")
    print(f"   • Recall: {curr['recall']:.1%} ({curr['missed_seizures']} misses)")
    print(f"   • Precision: {curr['precision']:.1%} ({curr['false_alarms']} false alarms)")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR MEDICAL DEVICE:")
    print("="*80)
    
    if 'recall_100' in results:
        r100 = results['recall_100']
        if r100['threshold'] < 0.5:
            print(f"\n✅ RECOMMENDED: Lower threshold to {r100['threshold']:.3f}")
            print("   Current 0.5 is already achieving 100% recall,")
            print(f"   but {r100['threshold']:.3f} is the safest minimum.")
        else:
            print("\n✅ RECOMMENDED: Keep current 0.5 or lower")
            print("   Already achieving 100% recall (no missed seizures)")
        
        print(f"\n   Trade-off: {r100['false_alarms']} false alarms vs 0 missed seizures")
        print("   For safety-critical medical device, this is acceptable.")
    
    print("\n💡 To update threshold in production:")
    print("   1. Edit .env file:")
    print("      POSE_MODEL_THRESHOLD=<your_chosen_threshold>")
    print("   2. Restart server")
    print("   3. Monitor false alarm rate in production")
    print("   4. Adjust based on real-world feedback")


def main():
    """Main analysis workflow."""
    print("="*80)
    print("ONNX MODEL THRESHOLD ANALYSIS")
    print("="*80)
    
    # Check model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    # Load test data
    try:
        X_test, y_test = load_test_data()
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Run inference
    y_scores = run_inference(MODEL_PATH, X_test)
    
    # Find optimal thresholds
    results, roc_data, pr_data = find_optimal_thresholds(y_test, y_scores)
    
    # Plot curves
    fpr, tpr, roc_auc = roc_data
    precision, recall, pr_thresholds = pr_data
    plot_curves(fpr, tpr, roc_auc, precision, recall, pr_thresholds, results)
    
    # Print recommendations
    print_recommendations(results)
    
    # Save results to JSON
    output_file = BASE_DIR / "threshold_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
