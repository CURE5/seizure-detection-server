#!/usr/bin/env python3
"""
Train hypermotor seizure detection model.

This script trains a hybrid LSTM model for patient-specific seizure detection
using extracted pose sequences and motion features.
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple
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
    print("❌ PyTorch not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        roc_auc_score, precision_recall_curve
    )
    print("✓ scikit-learn available")
except ImportError:
    print("❌ scikit-learn not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])


class PoseSequenceDataset(Dataset):
    """Dataset for pose sequences."""
    
    def __init__(self, sequences: List[Path], label: int):
        """
        Args:
            sequences: List of .npy file paths
            label: 0 for normal, 1 for seizure
        """
        self.sequences = sequences
        self.label = label
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Load sequence: (30, 33, 3)
        seq = np.load(self.sequences[idx])
        
        # Convert to tensor
        seq_tensor = torch.FloatTensor(seq)
        label_tensor = torch.FloatTensor([self.label])
        
        return seq_tensor, label_tensor


class HypermotorSeizureDetector(nn.Module):
    """
    Hybrid LSTM model for hypermotor seizure detection.
    
    Architecture:
    - Input: (batch, 30, 33, 3) = 30 frames × 33 keypoints × (x,y,visibility)
    - Flatten keypoints: (batch, 30, 99)
    - Bidirectional LSTM: (batch, 30, 256)
    - Attention mechanism
    - Dense layers with dropout
    - Output: (batch, 1) = seizure probability
    
    Uses all 33 MediaPipe landmarks for maximum fidelity:
    - Face detail (11 points): nose, eyes, ears, mouth
    - Hand detail (6 points): wrists, index, pinky, thumb
    - Foot detail (6 points): ankles, heels, toes
    - Core body (10 points): shoulders, elbows, hips, knees
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
            nn.Dropout(dropout),
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


def load_splits(splits_file: Path) -> Dict:
    """Load train/val/test splits."""
    with open(splits_file, 'r') as f:
        return json.load(f)


def create_dataloaders(
    data_dir: Path,
    splits: Dict,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    seizure_dir = data_dir / 'seizure' / 'processed'
    normal_dir = data_dir / 'normal' / 'processed'
    
    # Training data
    train_seizure = [
        seizure_dir / f for f in splits['train']['seizure']
    ]
    train_normal = [
        normal_dir / f for f in splits['train']['normal']
    ]
    
    train_dataset_sz = PoseSequenceDataset(train_seizure, label=1)
    train_dataset_norm = PoseSequenceDataset(train_normal, label=0)
    train_dataset = torch.utils.data.ConcatDataset([
        train_dataset_sz, train_dataset_norm
    ])
    
    # Validation data
    val_seizure = [
        seizure_dir / f for f in splits['validation']['seizure']
    ]
    val_normal = [
        normal_dir / f for f in splits['validation']['normal']
    ]
    
    val_dataset_sz = PoseSequenceDataset(val_seizure, label=1)
    val_dataset_norm = PoseSequenceDataset(val_normal, label=0)
    val_dataset = torch.utils.data.ConcatDataset([
        val_dataset_sz, val_dataset_norm
    ])
    
    # Test data
    test_seizure = [
        seizure_dir / f for f in splits['test']['seizure']
    ]
    test_normal = [
        normal_dir / f for f in splits['test']['normal']
    ]
    
    test_dataset_sz = PoseSequenceDataset(test_seizure, label=1)
    test_dataset_norm = PoseSequenceDataset(test_normal, label=0)
    test_dataset = torch.utils.data.ConcatDataset([
        test_dataset_sz, test_dataset_norm
    ])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\n📊 Dataset sizes:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"    - Seizure: {len(train_seizure)}")
    print(f"    - Normal: {len(train_normal)}")
    print(f"  Validation: {len(val_dataset)} sequences")
    print(f"    - Seizure: {len(val_seizure)}")
    print(f"    - Normal: {len(val_normal)}")
    print(f"  Test: {len(test_dataset)} sequences")
    print(f"    - Seizure: {len(test_seizure)}")
    print(f"    - Normal: {len(test_normal)}")
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in dataloader:
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
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    # Calculate recall (sensitivity) and precision
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return avg_loss, accuracy, recall, precision


def main():
    parser = argparse.ArgumentParser(
        description="Train hypermotor seizure detection model"
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('training_data'),
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--splits-file',
        type=Path,
        default=Path('training_data/dataset_splits.json'),
        help='Dataset splits file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models'),
        help='Directory to save trained model'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='LSTM hidden size'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    print(f"\n📂 Loading dataset splits from {args.splits_file}")
    splits = load_splits(args.splits_file)
    
    # Create dataloaders
    print("\n🔄 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        splits,
        batch_size=args.batch_size
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = HypermotorSeizureDetector(
        hidden_size=args.hidden_size
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("="*70)
    
    best_recall = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_recall, val_precision = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_recall)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"Recall: {val_recall:.4f}, Precision: {val_precision:.4f}")
        
        # Save best model (prioritize recall for seizure detection)
        if val_recall > best_recall:
            best_recall = val_recall
            best_epoch = epoch + 1
            
            model_path = args.output_dir / 'hypermotor_detector_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_recall': val_recall,
                'val_precision': val_precision,
                'val_accuracy': val_acc,
            }, model_path)
            
            print(f"  ✓ Saved best model (recall: {val_recall:.4f})")
    
    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"Best model: Epoch {best_epoch}, Recall: {best_recall:.4f}")
    
    # Test on hold-out set
    print("\n📊 Evaluating on test set...")
    model_path = args.output_dir / 'hypermotor_detector_best.pth'
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_recall, test_precision = validate(
        model, test_loader, criterion, device
    )
    
    print("\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Recall (Sensitivity): {test_recall:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    
    # Save final results
    results = {
        'best_epoch': best_epoch,
        'best_val_recall': float(best_recall),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_recall': float(test_recall),
        'test_precision': float(test_precision),
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = args.output_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    print("\n" + "="*70)
    print("Next step: Export to ONNX for deployment")
    print("  python scripts/export_to_onnx.py")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
