#!/usr/bin/env python3
"""
CIFAR-10 MobileNetV2 Baseline (FP32)

Train/fine-tune MobileNetV2 on CIFAR-10 without quantization.
Logs activation histograms and statistics to TensorBoard.

This serves as the baseline for comparing with BR-QAT.

Usage:
    python experiments/cifar10_mobilenet_baseline.py \
        --epochs 100 --batch-size 128 --lr 0.01 --gpu 0
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
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm


def get_mobilenetv2_cifar10(num_classes=10, pretrained=False):
    """
    Get MobileNetV2 adapted for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (default: 10)
        pretrained: Load ImageNet pretrained weights (default: False)
    
    Returns:
        Modified MobileNetV2 model for CIFAR-10
    """
    # Load pre-defined MobileNetV2 (using new weights API)
    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = mobilenet_v2(weights=weights)
    
    # Modify first conv layer for CIFAR-10 (32x32 input)
    # Original: stride=2, now: stride=1 for small images
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Modify classifier for CIFAR-10 (10 classes instead of 1000)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
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
    Collect and log activation histograms for all ReLU6 layers.
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
    
    # Register hooks on all ReLU6 layers
    hooks = []
    relu_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU6):
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
    parser = argparse.ArgumentParser(description='CIFAR-10 MobileNetV2 Baseline')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='Weight decay (default: 4e-5)')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("CIFAR-10 MobileNetV2 Baseline Training (FP32)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Pretrained: {args.pretrained}")
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
    log_dir = f'./runs/cifar10_mobilenet_baseline_{timestamp}'
    checkpoint_path = f'./checkpoints/cifar10_mobilenet_baseline_{timestamp}.pth'
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    
    # Model (using torchvision's pre-defined MobileNetV2)
    model = get_mobilenetv2_cifar10(num_classes=10, pretrained=args.pretrained).to(device)
    
    print(f"\nModel: MobileNetV2 (from torchvision)")
    if args.pretrained:
        print("  ✓ Loaded ImageNet pretrained weights!")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    print()
    
    # Optimizer and criterion
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler (cosine annealing with warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
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

