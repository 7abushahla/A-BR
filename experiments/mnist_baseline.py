"""
Baseline MNIST Training with TensorBoard Histogram Logging

This script trains a simple CNN on MNIST and logs activation histograms
to TensorBoard for visualization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import time
import json
import csv
import argparse
import random
from datetime import datetime
import numpy as np


class ClippedReLU(nn.Module):
    """ReLU with configurable clipping value (e.g., ReLU1, ReLU6)."""
    def __init__(self, clip_value=6.0):
        super().__init__()
        self.clip_value = clip_value
    
    def forward(self, x):
        return torch.clamp(x, 0.0, self.clip_value)


class PlainConvFlatten(nn.Module):
    """
    Simple CNN for MNIST - NO BatchNorm version.
    This matches the architecture of the regularized versions for fair comparison.
    """
    def __init__(self, input_channels=1, num_classes=10, base=16, clip_value=None):
        super().__init__()
        self.clip_value = clip_value
        
        # Choose activation function
        if clip_value is not None:
            activation_fn = lambda: ClippedReLU(clip_value)
        else:
            activation_fn = nn.ReLU
        
        # Conv layers with bias (no BN)
        self.conv1 = nn.Conv2d(input_channels, base, 3, stride=2, padding=1, bias=True)
        self.relu1 = activation_fn()
        
        self.conv2 = nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=True)
        self.relu2 = activation_fn()
        
        self.conv3 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=True)
        self.relu3 = activation_fn()
        
        self.conv4 = nn.Conv2d(base*4, base*4, 3, stride=1, padding=1, bias=True)
        self.relu4 = activation_fn()
        
        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(base*4 * 4 * 4, 128, bias=True)
        self.relu5 = activation_fn()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv blocks (no BN)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        # Classifier
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        
        return x


def get_data_loaders(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Split train into train/val (90/10)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def log_activation_histograms(model, data_loader, writer, epoch, device, csv_writer=None):
    """
    Log activation histograms for all ReLU/ClippedReLU layers.
    Returns kurtosis stats for printing.
    """
    model.eval()
    
    # Dictionary to store activations
    activations = {}
    
    # Hook function
    def get_activation(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, ClippedReLU)):
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Run forward pass on a batch
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            _ = model(images)
            break  # Just one batch for histogram
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Collect kurtosis values
    kurtosis_values = {}
    
    # Log histograms to TensorBoard
    for name, acts_list in activations.items():
        acts = torch.cat(acts_list, dim=0)
        writer.add_histogram(f'activations/{name}', acts, epoch)
        
        # Also log statistics
        writer.add_scalar(f'activation_stats/{name}/mean', acts.mean().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/std', acts.std().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/max', acts.max().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/min', acts.min().item(), epoch)
        
        # Compute kurtosis
        mean = acts.mean()
        std = acts.std()
        if std > 1e-8:
            standardized = (acts - mean) / std
            kurtosis = (standardized ** 4).mean().item()
            writer.add_scalar(f'activation_stats/{name}/kurtosis', kurtosis, epoch)
            kurtosis_values[name] = kurtosis
            
            # Write to CSV if provided
            if csv_writer is not None:
                csv_writer.writerow({
                    'epoch': epoch,
                    'layer': name,
                    'kurtosis': f'{kurtosis:.4f}',
                    'mean': f'{acts.mean().item():.4f}',
                    'std': f'{acts.std().item():.4f}',
                    'max': f'{acts.max().item():.4f}'
                })
    
    model.train()
    return kurtosis_values


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='MNIST Baseline Training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs (default: 40)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 128)')
    parser.add_argument('--clip-value', type=float, default=None, help='ReLU clip value (None=standard ReLU, 1.0=ReLU1, 6.0=ReLU6)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs
    batch_size = args.batch_size
    clip_value = args.clip_value
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*70)
    print("MNIST Baseline Training with TensorBoard Histograms")
    print("="*70)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Clip value: {clip_value if clip_value is not None else 'None (standard ReLU)'}")
    print(f"Run ID: {timestamp}\n")
    
    # Create model
    model = PlainConvFlatten(input_channels=1, num_classes=10, base=16, clip_value=clip_value)
    model = model.to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Data
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}\n")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # TensorBoard with timestamp
    log_dir = Path(f'./runs/mnist_baseline_{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    print("Run: tensorboard --logdir=runs\n")
    
    # CSV log file for kurtosis with timestamp
    csv_path = Path(f'./logs/mnist_baseline_{timestamp}_kurtosis.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'layer', 'kurtosis', 'mean', 'std', 'max'])
    csv_writer.writeheader()
    print(f"Kurtosis log: {csv_path}\n")
    
    # Log initial histograms
    print("Logging initial activation histograms...")
    kurtosis_stats = log_activation_histograms(model, val_loader, writer, 0, device, csv_writer)
    print(f"Initial kurtosis values:")
    if kurtosis_stats:
        for name, kurt in kurtosis_stats.items():
            print(f"  {name}: {kurt:.4f}")
    else:
        print("  (No kurtosis stats computed)")
    print()
    
    # Training loop
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Log to TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/val', val_acc, epoch)
        
        # Log activation histograms every 2 epochs
        if epoch % 2 == 0 or epoch == epochs:
            kurtosis_stats = log_activation_histograms(model, val_loader, writer, epoch, device, csv_writer)
            if kurtosis_stats:
                avg_kurtosis = sum(kurtosis_stats.values()) / len(kurtosis_stats)
                print(f"  → Avg Kurtosis: {avg_kurtosis:.4f} (target: 3.0 for Gaussian)")
            else:
                print(f"  → No kurtosis stats (std too low or no ReLU layers found)")
        
        # Print progress
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
    
    # Final test
    print("\n" + "="*70)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("="*70)
    
    # Log final test metrics to TensorBoard
    writer.add_scalar('loss/test', test_loss, epochs)
    writer.add_scalar('accuracy/test', test_acc, epochs)
    writer.flush()
    
    # Save model with timestamp
    save_path = Path(f'./checkpoints/mnist_baseline_{timestamp}.pth')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'timestamp': timestamp,
        'clip_value': clip_value,
        'seed': args.seed,
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Close files
    csv_file.close()
    writer.close()
    
    print(f"\n✓ Training complete!")
    print(f"  Kurtosis log: {csv_path}")
    print(f"  TensorBoard: tensorboard --logdir=runs")
    print(f"  View CSV: cat {csv_path}")


if __name__ == '__main__':
    main()

