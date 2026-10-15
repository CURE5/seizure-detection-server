#!/usr/bin/env python3
"""
Export trained motion model to ONNX format.

Usage:
    python scripts/export_motion_model_to_onnx.py
    
Requirements:
    pip install onnx onnxscript
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_motion_model import MotionSeizureDetector
from seizure_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


def export_model(
    pth_path: Path = Path("models/motion_seizure_detector_best.pth"),
    onnx_path: Path = Path("models/motion_seizure_detector.onnx"),
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
):
    """Export PyTorch model to ONNX format."""
    
    device = torch.device("cpu")  # Export on CPU for compatibility
    
    # Create model with same architecture
    logger.info("Creating model...")
    model = MotionSeizureDetector(
        input_size=3,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    
    # Load trained weights
    logger.info(f"Loading weights from {pth_path}")
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    # Export to ONNX
    logger.info(f"Exporting to {onnx_path}")
    dummy_input = torch.randn(1, 30, 3).to(device)
    
    # Use legacy exporter (dynamo=False) which is more stable for LSTM
    # Use opset 14 which has good LSTM support and is stable
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=False,  # Disable to avoid optimization issues
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Use legacy exporter
    )
    
    logger.info(f"✅ Model exported to {onnx_path}")
    
    # Verify the export
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model validated successfully")
    except ImportError:
        logger.warning("⚠️  onnx package not available for validation")
    except Exception as e:
        logger.error(f"❌ ONNX validation failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export motion model to ONNX")
    parser.add_argument("--pth", type=Path, default=Path("models/motion_seizure_detector_best.pth"))
    parser.add_argument("--onnx", type=Path, default=Path("models/motion_seizure_detector.onnx"))
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    args = parser.parse_args()
    
    export_model(args.pth, args.onnx, args.hidden_size, args.num_layers, args.dropout)
