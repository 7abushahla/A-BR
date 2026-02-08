#!/usr/bin/env python3
"""
CIFAR-10 Simple CNN Baseline (FP32, no quantization)

Same architecture as QAT version but without quantization.
This serves as the baseline for comparison.

Usage:
    # Default: Standard ReLU (no clipping)
    python experiments/cifar10_baseline.py \
        --epochs 100 --batch-size 256 --gpu 0
    
    # Optional: With clipping (for fairer comparison to QAT)
    python experiments/cifar10_baseline.py \
        --epochs 100 --batch-size 256 --clip-value 1.0 --gpu 0
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm


class ClippedReLU(nn.Module):
    """ReLU with configurable clipping value (e.g., ReLU1, ReLU6)."""
    def __init__(self, clip_value=6.0):
        super().__init__()
        self.clip_value = clip_value
    
    def forward(self, x):
        return torch.clamp(x, 0.0, self.clip_value)


class SimpleCNN_CIFAR10(nn.Module):
    """
    Simple CNN for CIFAR-10 (FP32 baseline).
    Same architecture as QAT version but without quantization.
    """
    
    def __init__(self, num_classes=10, base=32, clip_value=None):
        super().__init__()
        
        self.clip_value = clip_value
        
        # Choose activation function
        if clip_value is not None:
            activation_fn = lambda: ClippedReLU(clip_value)
        else:
            activation_fn = nn.ReLU
        
        # Input: 32x32x3
        self.conv1 = nn.Conv2d(3, base, 3, stride=2, padding=1, bias=True)
        self.relu1 = activation_fn()
        
        self.conv2 = nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=True)
        self.relu2 = activation_fn()
        
        self.conv3 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=True)
        self.relu3 = activation_fn()
        
        self.conv4 = nn.Conv2d(base*4, base*4, 3, stride=1, padding=1, bias=True)
        self.relu4 = activation_fn()
        
        # Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(base*4*4*4, 128, bias=True)
        self.relu5 = activation_fn()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        
        return x


def get_cifar10_loaders(batch_size=256, num_workers=4):
    """Get CIFAR-10 train and test loaders."""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def log_activation_histograms(writer, model, loader, device, epoch, num_batches=10):
    """Collect and log activation histograms."""
    model.eval()
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook
    
    # Register hooks on ReLU layers (both ClippedReLU and standard ReLU)
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (ClippedReLU, nn.ReLU)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Collect activations
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)
            _ = model(data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Log histograms
    for name, acts_list in activations.items():
        all_acts = torch.cat([a.flatten() for a in acts_list])
        writer.add_histogram(f'activations/{name}', all_acts, epoch)
        
        # Statistics
        writer.add_scalar(f'activation_stats/{name}/mean', all_acts.mean().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/std', all_acts.std().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/max', all_acts.max().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/min', all_acts.min().item(), epoch)
    
    model.train()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test_epoch(model, loader, criterion, device):
    """Test for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Simple CNN Baseline')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--clip-value', type=float, default=None, help='ReLU clip value (None = standard ReLU, 1.0 = ReLU1, etc.)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("CIFAR-10 Simple CNN Baseline (FP32)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Clip value: {args.clip_value if args.clip_value is not None else 'None (standard ReLU)'}")
    print("="*70)
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    log_dir = f'./runs/cifar10_simple_baseline_{timestamp}'
    checkpoint_path = f'./checkpoints/cifar10_simple_baseline_{timestamp}.pth'
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    
    # Model
    model = SimpleCNN_CIFAR10(
        num_classes=10,
        base=32,
        clip_value=args.clip_value
    ).to(device)
    
    print(f"\nModel: SimpleCNN_CIFAR10 (FP32 baseline)")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/test', test_acc, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        
        # Log histograms periodically (same as QAT version)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            log_activation_histograms(writer, model, test_loader, device, epoch)
        
        writer.flush()
        
        # Terminal output
        print(f"Epoch {epoch+1}/{args.epochs} (LR={current_lr:.6f}):")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_acc,
                'best_accuracy': best_acc,
                'clip_value': args.clip_value,
            }, checkpoint_path)
            print(f"  ✓ New best: {best_acc:.2f}%")
        
        print()
    
    writer.close()
    
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

