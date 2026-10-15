#!/usr/bin/env python3
"""
Train motion-based seizure detection model.

This script trains an LSTM model using motion features extracted from videos.
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import List, Tuple
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Checking dependencies...")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.model_selection import train_test_split
    print("✓ scikit-learn available")
except ImportError:
    print("❌ scikit-learn not installed. Install with: pip install scikit-learn")
    sys.exit(1)

try:
    from tqdm import tqdm
    print("✓ tqdm available")
except ImportError:
    print("❌ tqdm not installed. Install with: pip install tqdm")
    sys.exit(1)

from seizure_detection.utils.logger import setup_logger

logger = setup_logger(__name__)


class MotionSequenceDataset(Dataset):
    """Dataset for motion feature sequences."""
    
    def __init__(self, sequences: List[Path], label: int):
        """
        Args:
            sequences: List of .npy file paths containing motion features
            label: 0 for normal, 1 for seizure
        """
        self.sequences = sequences
        self.label = label
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Load sequence: (30, 3) = 30 frames × 3 features (magnitude, density, direction)
        seq = np.load(self.sequences[idx])
        
        # Convert to tensor
        seq_tensor = torch.FloatTensor(seq)
        label_tensor = torch.FloatTensor([self.label])
        
        return seq_tensor, label_tensor


class MotionSeizureDetector(nn.Module):
    """
    LSTM model for motion-based seizure detection.
    
    Architecture:
    - Input: (batch, 30, 3) = 30 frames × 3 motion features
    - Bidirectional LSTM with attention
    - Dense layers with dropout
    - Output: (batch, 1) = seizure probability
    """
    
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, dropout=0.3):
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
        # x: (batch, 30, 3)
        
        # LSTM: (batch, 30, 256)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights: (batch, 30, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum: (batch, 256)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification: (batch, 1)
        output = self.classifier(context)
        
        return output


def load_data(data_dir: Path, 
              test_split: float = 0.15,
              val_split: float = 0.15) -> Tuple[List, List, List, List, List, List]:
    """
    Load and split motion feature sequences.
    
    Returns:
        (train_seizure, train_normal, val_seizure, val_normal, test_seizure, test_normal)
    """
    seizure_dir = data_dir / "seizure"
    normal_dir = data_dir / "normal"
    
    # Load all sequences
    seizure_files = sorted(list(seizure_dir.glob("*.npy")))
    normal_files = sorted(list(normal_dir.glob("*.npy")))
    
    logger.info(f"Found {len(seizure_files)} seizure sequences")
    logger.info(f"Found {len(normal_files)} normal sequences")
    
    if len(seizure_files) == 0 or len(normal_files) == 0:
        raise ValueError("No sequences found! Run extract_motion_features.py first")
    
    # Split seizure data
    sz_train_val, sz_test = train_test_split(
        seizure_files, test_size=test_split, random_state=42
    )
    sz_train, sz_val = train_test_split(
        sz_train_val, test_size=val_split/(1-test_split), random_state=42
    )
    
    # Split normal data
    norm_train_val, norm_test = train_test_split(
        normal_files, test_size=test_split, random_state=42
    )
    norm_train, norm_val = train_test_split(
        norm_train_val, test_size=val_split/(1-test_split), random_state=42
    )
    
    logger.info(f"Split: Train={len(sz_train)+len(norm_train)}, "
               f"Val={len(sz_val)+len(norm_val)}, Test={len(sz_test)+len(norm_test)}")
    
    return sz_train, norm_train, sz_val, norm_val, sz_test, norm_test


def create_dataloaders(sz_train, norm_train, sz_val, norm_val, sz_test, norm_test,
                      batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    # Training dataset
    train_dataset = torch.utils.data.ConcatDataset([
        MotionSequenceDataset(sz_train, label=1),
        MotionSequenceDataset(norm_train, label=0)
    ])
    
    # Validation dataset
    val_dataset = torch.utils.data.ConcatDataset([
        MotionSequenceDataset(sz_val, label=1),
        MotionSequenceDataset(norm_val, label=0)
    ])
    
    # Test dataset
    test_dataset = torch.utils.data.ConcatDataset([
        MotionSequenceDataset(sz_test, label=1),
        MotionSequenceDataset(norm_test, label=0)
    ])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(train_loader, desc="Training", leave=False):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, data_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(data_loader), np.array(all_preds), np.array(all_probs), np.array(all_labels)


def export_to_onnx(model, output_path: Path, device):
    """Export model to ONNX format."""
    model.eval()
    
    # Dummy input: (1, 30, 3)
    dummy_input = torch.randn(1, 30, 3).to(device)
    
    # Use legacy exporter (dynamo=False) which is more stable for LSTM
    # Use opset 14 which has good LSTM support and is stable
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
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
    
    logger.info(f"✅ Model exported to {output_path}")


def save_checkpoint(checkpoint_path: Path, epoch: int, model_state: dict, 
                   optimizer_state: dict, scheduler_state: dict,
                   best_val_loss: float, patience_counter: int):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"💾 Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path, model, optimizer, scheduler, device):
    """Load training checkpoint."""
    if not checkpoint_path.exists():
        return 0, float('inf'), 0
    
    logger.info(f"💾 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint['patience_counter']
    
    logger.info(f"✅ Resumed from epoch {epoch+1}")
    return epoch + 1, best_val_loss, patience_counter


def main():
    parser = argparse.ArgumentParser(description="Train motion-based seizure detection model")
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="Directory containing extracted motion features")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=128,
                       help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--resume", type=Path, default=None,
                       help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Motion-Based Seizure Detection Training")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\n📂 Loading data...")
    sz_train, norm_train, sz_val, norm_val, sz_test, norm_test = load_data(args.data_dir)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        sz_train, norm_train, sz_val, norm_val, sz_test, norm_test,
        batch_size=args.batch_size
    )
    
    # Create model
    logger.info("\n🧠 Creating model...")
    model = MotionSeizureDetector(
        input_size=3,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Load checkpoint if resuming
    checkpoint_path = args.output_dir / "training_checkpoint.pth"
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if args.resume:
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
    elif checkpoint_path.exists():
        logger.info(f"Found existing checkpoint: {checkpoint_path}")
        resume = input("Resume from checkpoint? (y/n): ").lower().strip()
        if resume == 'y':
            start_epoch, best_val_loss, patience_counter = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
    
    # Training loop
    logger.info("\n🏋️ Training...")
    best_model_state = model.state_dict().copy() if start_epoch == 0 else None
    max_patience = 15
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_preds, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Calculate metrics
        val_acc = np.mean(val_preds == val_labels)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model and checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"  ✓ New best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint
            save_checkpoint(
                checkpoint_path, epoch, best_model_state,
                optimizer.state_dict(), scheduler.state_dict(),
                best_val_loss, patience_counter
            )
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    logger.info("\n📊 Evaluating on test set...")
    test_loss, test_preds, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    
    # Metrics
    test_acc = np.mean(test_preds == test_labels)
    test_auc = roc_auc_score(test_labels, test_probs)
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {test_acc:.4f}")
    logger.info(f"AUC-ROC: {test_auc:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Normal', 'Seizure']))
    logger.info("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    
    # Save PyTorch model
    torch_path = args.output_dir / "motion_seizure_detector_best.pth"
    torch.save(best_model_state, torch_path)
    logger.info(f"\n💾 PyTorch model saved: {torch_path}")
    
    # Export to ONNX (optional - skip if onnxscript not available)
    onnx_path = args.output_dir / "motion_seizure_detector.onnx"
    try:
        export_to_onnx(model, onnx_path, device)
    except ImportError as e:
        logger.warning(f"⚠️  ONNX export skipped (missing dependency): {e}")
        logger.info("   To enable ONNX export, install: pip install onnx onnxscript")
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        logger.info(f"   Model is still saved as PyTorch format: {torch_path}")
    
    # Save metadata
    metadata = {
        "training_date": str(datetime.now()),
        "data_dir": str(args.data_dir),
        "epochs_trained": epoch + 1,
        "best_val_loss": float(best_val_loss),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "model_params": {
            "input_size": 3,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "training_params": {
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        }
    }
    
    metadata_path = args.output_dir / "motion_model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"📝 Metadata saved: {metadata_path}")
    
    # Clean up checkpoint file after successful training
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("🧹 Cleaned up checkpoint file")
    
    logger.info("\n✅ Training complete!")
    logger.info("\n💡 Note: Training can be interrupted (Ctrl+C) and resumed with:")
    logger.info(f"   python scripts/train_motion_model.py --data-dir {args.data_dir} --resume {checkpoint_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
