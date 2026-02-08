#!/usr/bin/env python3
"""
CIFAR-10 ResNet18 Baseline (FP32)

Train/fine-tune ResNet18 on CIFAR-10 without quantization.
Logs activation histograms and statistics to TensorBoard.

This serves as the baseline for comparing with BR-QAT.

Key Features:
- Supports ImageNet pretrained weights (transfer learning)
- Can train from scratch on CIFAR-10
- ReLU layers are easily replaceable (for future ClippedReLU/ReLU1/ReLU6)
- Adapted for CIFAR-10's 32x32 input size

Usage:
    # Train from scratch
    python experiments/cifar10_resnet18_baseline.py \
        --epochs 200 --batch-size 128 --lr 0.1 --gpu 0

    # Fine-tune from ImageNet pretrained
    python experiments/cifar10_resnet18_baseline.py \
        --pretrained --epochs 100 --batch-size 128 --lr 0.01 --gpu 0
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm


class ClippedReLU(nn.Module):
    """
    ReLU with configurable clipping value.
    
    This is provided for future use (e.g., ReLU1, ReLU6).
    Default clip_value=6.0 matches MobileNetV2's ReLU6.
    """
    def __init__(self, clip_value=6.0):
        super().__init__()
        self.clip_value = clip_value
    
    def forward(self, x):
        return torch.clamp(x, 0.0, self.clip_value)
    
    def __repr__(self):
        return f'ClippedReLU(clip_value={self.clip_value})'


def replace_relu_with_clipped(model, clip_value=6.0):
    """
    Replace all ReLU layers with ClippedReLU.
    
    This is optional - by default we use standard ReLU.
    Call this function AFTER loading the model if you want clipped activations.
    
    Args:
        model: PyTorch model
        clip_value: Clipping threshold (e.g., 1.0 for ReLU1, 6.0 for ReLU6)
    
    Returns:
        Modified model with ClippedReLU layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            # Replace ReLU with ClippedReLU
            setattr(model, name, ClippedReLU(clip_value=clip_value))
        else:
            # Recursively replace in child modules
            replace_relu_with_clipped(module, clip_value)
    return model


def get_resnet18_cifar10(num_classes=10, pretrained=False, clip_value=None):
    """
    Get ResNet18 adapted for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (default: 10)
        pretrained: Load ImageNet pretrained weights (default: False)
        clip_value: If not None, replace ReLU with ClippedReLU(clip_value)
                   Examples: 1.0 for ReLU1, 6.0 for ReLU6
    
    Returns:
        Modified ResNet18 model for CIFAR-10
    """
    # Load ResNet18 (using new weights API)
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        print("  Loading ImageNet pretrained weights...")
    else:
        weights = None
    
    model = resnet18(weights=weights)
    
    # ============================================================
    # CIFAR-10 Adaptations (32x32 input instead of 224x224)
    # ============================================================
    
    # 1. Modify first conv layer: 7x7 stride=2 -> 3x3 stride=1
    #    Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    #    CIFAR-10: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    #    This prevents aggressive downsampling of small 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove MaxPool layer (too aggressive for 32x32 images)
    #    Original: MaxPool2d(kernel_size=3, stride=2, padding=1)
    #    CIFAR-10: Identity (no pooling)
    model.maxpool = nn.Identity()
    
    # 3. Modify final FC layer: 1000 classes -> 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 4. Optionally replace ReLU with ClippedReLU
    if clip_value is not None:
        print(f"  Replacing ReLU with ClippedReLU(clip_value={clip_value})...")
        model = replace_relu_with_clipped(model, clip_value=clip_value)
    
    return model


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """Get CIFAR-10 train and test loaders with data augmentation."""
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for test
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
    """
    Collect and log activation histograms for all ReLU/ClippedReLU layers.
    """
    model.eval()
    
    # Storage for activations
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook
    
    # Register hooks on all ReLU layers (including ClippedReLU)
    hooks = []
    relu_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, ClippedReLU)):
            relu_count += 1
            hook_name = f"relu{relu_count}"
            hooks.append(module.register_forward_hook(make_hook(hook_name)))
    
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
        
        # Log statistics
        mean_val = all_acts.mean().item()
        std_val = all_acts.std().item()
        max_val = all_acts.max().item()
        min_val = all_acts.min().item()
        
        writer.add_scalar(f'activation_stats/{name}/mean', mean_val, epoch)
        writer.add_scalar(f'activation_stats/{name}/std', std_val, epoch)
        writer.add_scalar(f'activation_stats/{name}/max', max_val, epoch)
        writer.add_scalar(f'activation_stats/{name}/min', min_val, epoch)
        
        # Compute kurtosis (measures tail heaviness / peakedness)
        if std_val > 1e-8:
            standardized = (all_acts - mean_val) / std_val
            kurtosis = (standardized ** 4).mean().item()
            writer.add_scalar(f'activation_stats/{name}/kurtosis', kurtosis, epoch)
    
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
    parser = argparse.ArgumentParser(description='CIFAR-10 ResNet18 Baseline')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1 for scratch, 0.01 for pretrained)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--clip-value', type=float, default=None, 
                       help='Replace ReLU with ClippedReLU (e.g., 1.0 for ReLU1, 6.0 for ReLU6, None for standard ReLU)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("CIFAR-10 ResNet18 Baseline Training (FP32)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Pretrained: {args.pretrained}")
    if args.clip_value is not None:
        print(f"Activation: ClippedReLU(clip_value={args.clip_value})")
    else:
        print(f"Activation: ReLU (standard)")
    print("="*70)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    log_dir = f'./runs/cifar10_resnet18_baseline_{timestamp}'
    checkpoint_path = f'./checkpoints/cifar10_resnet18_baseline_{timestamp}.pth'
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    
    # Model
    model = get_resnet18_cifar10(
        num_classes=10, 
        pretrained=args.pretrained,
        clip_value=args.clip_value
    ).to(device)
    
    print(f"\nModel: ResNet18 (torchvision, adapted for CIFAR-10)")
    if args.pretrained:
        print("  ✓ Loaded ImageNet pretrained weights!")
        print("  ✓ Modified conv1 (7x7,s=2 -> 3x3,s=1) and removed maxpool")
        print("  ✓ Modified FC layer (1000 -> 10 classes)")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    print()
    
    # Count ReLU layers
    relu_count = sum(1 for m in model.modules() if isinstance(m, (nn.ReLU, ClippedReLU)))
    print(f"Total ReLU layers: {relu_count}")
    print()
    
    # Optimizer and criterion
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    # For pretrained: CosineAnnealing (smooth decay)
    # For scratch: MultiStepLR (standard ResNet schedule)
    if args.pretrained:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        print("LR Scheduler: CosineAnnealingLR (for pretrained)")
    else:
        # Standard CIFAR-10 schedule: decay at 50%, 75% of training
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        print(f"LR Scheduler: MultiStepLR (decay at epochs {milestones})")
    print()
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    print("="*70)
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
        
        # Log activation histograms periodically
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  Logging activation histograms...")
            log_activation_histograms(writer, model, test_loader, device, epoch)
        
        writer.flush()
        
        # Terminal output
        print(f"Epoch {epoch+1}/{args.epochs} (LR={current_lr:.6f}):")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'best_accuracy': best_acc,
                'pretrained': args.pretrained,
                'clip_value': args.clip_value,
                'seed': args.seed,
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

