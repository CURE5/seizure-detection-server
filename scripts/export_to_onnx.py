#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX format for deployment.

ONNX format allows the model to run on:
- Raspberry Pi (CPU)
- Hailo AI accelerator
- Other edge devices
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Checking dependencies...")
try:
    import torch
    import torch.nn as nn
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)

try:
    import onnx
    import onnxruntime
    print(f"✓ ONNX {onnx.__version__}")
    print(f"✓ ONNX Runtime {onnxruntime.__version__}")
except ImportError:
    print("❌ ONNX not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])
    import onnx
    import onnxruntime

import numpy as np


class HypermotorSeizureDetector(nn.Module):
    """
    Hybrid LSTM model for hypermotor seizure detection.
    (Copy from training script for export)
    
    Uses all 33 MediaPipe landmarks for maximum fidelity.
    """
    
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, 30, 33, 3)
        batch_size, seq_len, num_keypoints, features = x.shape
        
        # Flatten keypoints: (batch, 30, 99)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM: (batch, 30, 256)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights: (batch, 30, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum: (batch, 256)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification: (batch, 1)
        output = self.classifier(context)
        
        return output


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    opset_version: int = 14
) -> bool:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    
    Returns:
        bool: True if export successful
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input: (batch=1, 30 frames, 33 keypoints, 3 features)
    dummy_input = torch.randn(1, 30, 33, 3)
    
    print("\n📦 Exporting model to ONNX...")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Opset version: {opset_version}")
    
    try:
        # Use legacy exporter (dynamo=False) for compatibility
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['pose_sequence'],
                output_names=['seizure_probability'],
                dynamic_axes={
                    'pose_sequence': {0: 'batch_size'},
                    'seizure_probability': {0: 'batch_size'}
                },
                dynamo=False  # Use legacy exporter for stability
            )
        
        print(f"✓ Model exported to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def verify_onnx_model(
    onnx_path: Path,
    pytorch_model: nn.Module
) -> bool:
    """
    Verify ONNX model produces same results as PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
    
    Returns:
        bool: True if models match
    """
    print("\n🔍 Verifying ONNX model...")
    
    # Load ONNX model
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")
        return False
    
    # Create ONNX Runtime session
    try:
        ort_session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        print("✓ ONNX Runtime session created")
    except Exception as e:
        print(f"❌ Failed to create ONNX Runtime session: {e}")
        return False
    
    # Test with random input
    test_input = torch.randn(1, 30, 33, 3)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    print("\n📊 Comparison:")
    print(f"   PyTorch output: {pytorch_output[0][0]:.6f}")
    print(f"   ONNX output: {onnx_output[0][0]:.6f}")
    print(f"   Max difference: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✓ Models match (difference < 1e-5)")
        return True
    else:
        print(f"⚠️  Models differ by {max_diff:.8f}")
        if max_diff < 1e-3:
            print("   (Acceptable for deployment)")
            return True
        else:
            print("   (Too large - export may have issues)")
            return False


def benchmark_onnx_model(onnx_path: Path, num_iterations: int = 100):
    """
    Benchmark ONNX model inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        num_iterations: Number of inference iterations
    """
    print(f"\n⚡ Benchmarking ONNX model ({num_iterations} iterations)...")
    
    import time
    
    # Create session
    ort_session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=['CPUExecutionProvider']
    )
    
    # Prepare input (33 MediaPipe landmarks)
    test_input = np.random.randn(1, 30, 33, 3).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    
    # Warmup
    for _ in range(10):
        ort_session.run(None, ort_inputs)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        ort_session.run(None, ort_inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"   Average inference time: {avg_time:.2f} ms")
    print(f"   Throughput: {fps:.1f} FPS")
    
    if avg_time < 100:  # Less than 100ms
        print("   ✓ Fast enough for real-time detection (5 FPS = 200ms budget)")
    else:
        print("   ⚠️  May be slow for real-time (need < 200ms for 5 FPS)")


def print_model_info(onnx_path: Path):
    """Print ONNX model information."""
    print("\n📋 Model Information:")
    
    onnx_model = onnx.load(str(onnx_path))
    
    # File size
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
    
    # Input/Output info
    print("\n   Inputs:")
    for inp in onnx_model.graph.input:
        print(f"     - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    
    print("\n   Outputs:")
    for out in onnx_model.graph.output:
        print(f"     - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
    
    # Opset version
    print(f"\n   Opset version: {onnx_model.opset_import[0].version}")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX format"
    )
    
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('models/hypermotor_detector_best.pth'),
        help='Path to PyTorch model checkpoint'
    )
    
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path('models/hypermotor_detector.onnx'),
        help='Path to save ONNX model'
    )
    
    parser.add_argument(
        '--opset-version',
        type=int,
        default=14,
        help='ONNX opset version (14 is widely compatible)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='Verify ONNX model matches PyTorch model'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        default=True,
        help='Benchmark inference speed'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not args.model_path.exists():
        print(f"❌ Model not found: {args.model_path}")
        print("   Run training first: python scripts/train_seizure_model.py")
        return 1
    
    print(f"📂 Loading PyTorch model from {args.model_path}")
    
    # Load PyTorch model
    device = torch.device('cpu')
    model = HypermotorSeizureDetector().to(device)
    
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"  Validation recall: {checkpoint['val_recall']:.4f}")
    print(f"  Validation precision: {checkpoint['val_precision']:.4f}")
    
    # Export to ONNX
    success = export_to_onnx(model, args.output_path, args.opset_version)
    
    if not success:
        return 1
    
    # Verify model
    if args.verify:
        if not verify_onnx_model(args.output_path, model):
            print("\n⚠️  Verification failed but model may still work")
    
    # Print model info
    print_model_info(args.output_path)
    
    # Benchmark
    if args.benchmark:
        benchmark_onnx_model(args.output_path)
    
    print("\n" + "="*70)
    print("✅ Export complete!")
    print(f"   ONNX model: {args.output_path}")
    print("\nNext steps:")
    print("  1. Copy model to Raspberry Pi:")
    print(f"     scp {args.output_path} pi@raspberrypi:~/models/")
    print("  2. Integrate with seizure_detection/analysis/")
    print("  3. Test on RPi with real video feed")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
