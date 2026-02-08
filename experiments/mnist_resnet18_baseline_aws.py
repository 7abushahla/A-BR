#!/usr/bin/env python3
"""
MNIST ResNet18 Baseline (FP32) - AWS/Slurm Optimized

Train/fine-tune ResNet18 on MNIST without quantization.
Optimized for batch jobs with proper logging.

Key Features:
- Supports ImageNet pretrained weights (transfer learning)
- Can train from scratch on MNIST
- ReLU layers are easily replaceable (for future ClippedReLU/ReLU1/ReLU6)
- Adapted for MNIST's 28x28 grayscale input
- Proper logging to file + stdout (for slurm)
- No histograms or plotting (faster, less memory)

Usage:
    # Train from scratch
    python experiments/mnist_resnet18_baseline_aws.py \
        --epochs 50 --batch-size 128 --lr 0.1 \
        --log-file logs/mnist_resnet18_baseline.log --gpu 0

    # Fine-tune from ImageNet pretrained
    python experiments/mnist_resnet18_baseline_aws.py \
        --pretrained --epochs 30 --batch-size 128 --lr 0.01 \
        --log-file logs/mnist_resnet18_baseline_pretrained.log --gpu 0
"""

import sys
import os
import logging
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


def get_logger(log_file: str, name: str = "mnist_resnet18_baseline") -> logging.Logger:
    """
    Setup logger that writes to both file and stdout (for slurm).
    
    Args:
        log_file: Path to log file
        name: Logger name
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if script is re-run
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Stream handler (stdout -> slurm-XXXX.out)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    return logger


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


def get_resnet18_mnist(num_classes=10, pretrained=False, clip_value=None, logger=None):
    """
    Get ResNet18 adapted for MNIST.
    
    Args:
        num_classes: Number of output classes (default: 10)
        pretrained: Load ImageNet pretrained weights (default: False)
        clip_value: If not None, replace ReLU with ClippedReLU(clip_value)
                   Examples: 1.0 for ReLU1, 6.0 for ReLU6
        logger: Logger instance
    
    Returns:
        Modified ResNet18 model for MNIST
    """
    log = logger.info if logger else print
    
    # Load ResNet18 (using new weights API)
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        log("  Loading ImageNet pretrained weights...")
    else:
        weights = None
    
    model = resnet18(weights=weights)
    
    # ============================================================
    # MNIST Adaptations (28x28 grayscale input instead of 224x224 RGB)
    # ============================================================
    
    # 1. Modify first conv layer: 3 channels (RGB) -> 1 channel (grayscale)
    #    Also: 7x7 stride=2 -> 3x3 stride=1 for small 28x28 images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove MaxPool layer (too aggressive for 28x28 images)
    model.maxpool = nn.Identity()
    
    # 3. Modify final FC layer: 1000 classes -> 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 4. Optionally replace ReLU with ClippedReLU
    if clip_value is not None:
        log(f"  Replacing ReLU with ClippedReLU(clip_value={clip_value})...")
        model = replace_relu_with_clipped(model, clip_value=clip_value)
    
    return model


def get_mnist_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """Get MNIST train and test loaders."""
    
    # Simple transforms for MNIST (grayscale, 28x28)
    # Note: MNIST mean=0.1307, std=0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device, logger=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
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
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test_epoch(model, loader, criterion, device, logger=None):
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
    parser = argparse.ArgumentParser(description='MNIST ResNet18 Baseline (AWS/Slurm)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for MNIST data')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1 for scratch, 0.01 for pretrained)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--clip-value', type=float, default=None, 
                       help='Replace ReLU with ClippedReLU (e.g., 1.0 for ReLU1, 6.0 for ReLU6, None for standard ReLU)')
    parser.add_argument('--log-file', type=str, default='logs/mnist_resnet18_baseline.log', help='Log file path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Setup logger
    log_file = os.path.abspath(args.log_file)
    logger = get_logger(log_file)
    
    logger.info("="*70)
    logger.info("MNIST ResNet18 Baseline Training (FP32) - AWS/Slurm")
    logger.info("="*70)
    logger.info(f"Arguments: {vars(args)}")
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Pretrained: {args.pretrained}")
    if args.clip_value is not None:
        logger.info(f"Activation: ClippedReLU(clip_value={args.clip_value})")
    else:
        logger.info(f"Activation: ReLU (standard)")
    logger.info("="*70)
    
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
    log_dir = f'./runs/mnist_resnet18_baseline_{timestamp}'
    checkpoint_path = f'./checkpoints/mnist_resnet18_baseline_{timestamp}.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Data
    logger.info(f"Loading MNIST dataset from: {args.data_dir}")
    train_loader, test_loader = get_mnist_loaders(args.batch_size, data_dir=args.data_dir)
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    model = get_resnet18_mnist(
        num_classes=10, 
        pretrained=args.pretrained,
        clip_value=args.clip_value,
        logger=logger
    ).to(device)
    
    logger.info(f"\nModel: ResNet18 (adapted for MNIST)")
    if args.pretrained:
        logger.info("  ✓ Loaded ImageNet pretrained weights!")
        logger.info("  ✓ Modified conv1 (3ch RGB -> 1ch grayscale, 7x7,s=2 -> 3x3,s=1) and removed maxpool")
        logger.info("  ✓ Modified FC layer (1000 -> 10 classes)")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params:,}")
    
    # Count ReLU layers
    relu_count = sum(1 for m in model.modules() if isinstance(m, (nn.ReLU, ClippedReLU)))
    logger.info(f"Total ReLU layers: {relu_count}")
    
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
        logger.info("LR Scheduler: CosineAnnealingLR (for pretrained)")
    else:
        # Standard MNIST schedule: decay at 50%, 75% of training
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        logger.info(f"LR Scheduler: MultiStepLR (decay at epochs {milestones})")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs: {log_dir}")
    
    logger.info("="*70)
    logger.info("Starting Training...")
    logger.info("="*70)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, logger)
        
        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device, logger)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/test', test_acc, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        writer.flush()
        
        # Terminal/log output
        logger.info(f"Epoch {epoch+1}/{args.epochs} (LR={current_lr:.6f}):")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
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
            logger.info(f"  ✓ New best: {best_acc:.2f}%")
    
    writer.close()
    
    logger.info("="*70)
    logger.info("Training Complete!")
    logger.info("="*70)
    logger.info(f"Best Test Accuracy: {best_acc:.2f}%")
    logger.info(f"Model saved to: {checkpoint_path}")
    logger.info(f"Logs written to: {log_file}")
    logger.info(f"TensorBoard logs: {log_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

