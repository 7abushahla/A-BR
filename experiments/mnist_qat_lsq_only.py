#!/usr/bin/env python3
"""
QAT with LSQ ONLY (No Bin Regularization)

This is to compare:
- LSQ QAT alone: Just fake quantization, no regularization pushing activations to bins
- LSQ QAT + BR: Fake quantization + bin regularization

Compare histograms to see if BR actually changes activation distributions!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from abr.lsq_quantizer import LSQ_ActivationQuantizer
from abr.hooks import ActivationHookManager


class QuantizedReLU(nn.Module):
    """Standard ReLU with LSQ quantization (no clipping)."""
    
    def __init__(self, num_bits=2):
        super().__init__()
        self.num_bits = num_bits
        self.quantization_enabled = False  # Start disabled for warmup
        
        # LSQ quantizer (will set clip_value very high to effectively disable clipping)
        self.quantizer = LSQ_ActivationQuantizer(
            num_bits=num_bits,
            clip_value=1000.0  # Very high, essentially no clipping
        )
    
    def forward(self, x):
        # Step 1: Standard ReLU (no clipping)
        x = torch.relu(x)
        
        # Step 2: LSQ quantization (only if enabled)
        if self.quantization_enabled:
            x = self.quantizer(x)
        
        return x
    
    def enable_quantization(self):
        """Enable quantization for QAT."""
        self.quantization_enabled = True
    
    def disable_quantization(self):
        """Disable quantization for FP32 training."""
        self.quantization_enabled = False


class PlainConvFlattenQAT(nn.Module):
    """Conv model with LSQ quantization on standard ReLU (NO clipping, NO BR)."""
    
    def __init__(self, input_channels=1, num_classes=10, base=16, num_bits=2):
        super().__init__()
        
        self.num_bits = num_bits
        
        # Conv blocks with Quantized Standard ReLU
        self.conv1 = nn.Conv2d(input_channels, base, 3, stride=2, padding=1, bias=True)
        self.relu1 = QuantizedReLU(num_bits)
        
        self.conv2 = nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=True)
        self.relu2 = QuantizedReLU(num_bits)
        
        self.conv3 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=True)
        self.relu3 = QuantizedReLU(num_bits)
        
        self.conv4 = nn.Conv2d(base*4, base*4, 3, stride=1, padding=1, bias=True)
        self.relu4 = QuantizedReLU(num_bits)
        
        # Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(base*4*4*4, 128, bias=True)
        self.relu5 = QuantizedReLU(num_bits)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
    
    def forward(self, x):
        x = x.float() / 255.0
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        
        return x


def log_activation_histograms(writer, hook_manager, epoch):
    """Log activation histograms to TensorBoard."""
    activations = hook_manager.get_activations()
    
    for name, acts in activations.items():
        acts_flat = acts.flatten()
        writer.add_histogram(f'activations/{name}', acts_flat, epoch)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch=0, debug=False):
    """Train for one epoch (NO BR, just task loss)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Only task loss (NO BR!)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Debug: print alpha values after first batch of first epoch
        if debug and epoch == 1 and batch_idx == 0:
            print("\n  Alpha values after first update:")
            for name, module in model.named_modules():
                if hasattr(module, 'quantizer') and hasattr(module.quantizer, 'alpha'):
                    print(f"    {name}: alpha={module.quantizer.alpha.item():.6f}, grad={module.quantizer.alpha.grad.item() if module.quantizer.alpha.grad is not None else 'None'}")
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='QAT with LSQ only (no BR, no clipping)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-bits', type=int, default=2)
    parser.add_argument('--warmup-epochs', type=int, default=5, 
                        help='Number of epochs to pre-train before enabling quantizers')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    import random
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("="*70)
    print("QAT with LSQ ONLY (No BR, No Clipping)")
    print("="*70)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Epochs: {args.epochs} (warmup: {args.warmup_epochs}, QAT: {args.epochs - args.warmup_epochs})")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Target bit-width: {args.num_bits} bits ({2**args.num_bits} levels)")
    print(f"Activation: Standard ReLU (no clipping)")
    print(f"Regularization: NONE (pure LSQ)")
    print("\nTraining strategy:")
    print(f"  Phase 1 (Epochs 1-{args.warmup_epochs}): FP32 warmup (quantizers disabled)")
    print(f"  Phase 2 (Epochs {args.warmup_epochs+1}-{args.epochs}): QAT with LSQ (quantizers enabled)")
    print("="*70)
    
    # Create model
    model = PlainConvFlattenQAT(
        input_channels=1,
        num_classes=10,
        base=16,
        num_bits=args.num_bits
    )
    model.to(device)
    
    # Print initial scales (note: quantizers are disabled during warmup)
    print("\nInitial quantization scales (will be used after warmup):")
    for name, module in model.named_modules():
        if isinstance(module, QuantizedReLU):
            alpha = module.quantizer.alpha.item()
            status = "DISABLED" if not module.quantization_enabled else "ENABLED"
            print(f"  {name}: alpha={alpha:.4f} [{status}]")
    
    # Hook manager for visualization (all layers)
    hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedReLU],
        layer_names=['relu1', 'relu2', 'relu3', 'relu4', 'relu5'],
        exclude_first_last=False,
        detach_activations=True  # No gradients needed
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'./runs/mnist_qat_lsq_only_{timestamp}'
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Enable quantizers after warmup
        if epoch == args.warmup_epochs + 1:
            print("\n" + "="*70)
            print(f"ENABLING QUANTIZERS (Epoch {epoch})")
            print("="*70)
            for name, module in model.named_modules():
                if isinstance(module, QuantizedReLU):
                    module.enable_quantization()
            print("Quantization enabled for all layers!")
            print("="*70 + "\n")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch, debug=(epoch == args.warmup_epochs + 1))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Log to TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/test', test_acc, epoch)
        
        # Log quantization scales
        for name, module in model.named_modules():
            if isinstance(module, QuantizedReLU):
                alpha = module.quantizer.alpha.item()
                writer.add_scalar(f'quant_scales/{name}', alpha, epoch)
        
        # Log histograms periodically
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs:
            log_activation_histograms(writer, hook_manager, epoch)
        
        writer.flush()
        
        # Print progress
        phase = "Warmup (FP32)" if epoch <= args.warmup_epochs else "QAT (Quantized)"
        print(f"Epoch {epoch}/{args.epochs} [{phase}]:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    # Print final scales
    print("\n" + "="*70)
    print("Final quantization scales:")
    print("="*70)
    for name, module in model.named_modules():
        if isinstance(module, QuantizedReLU):
            alpha = module.quantizer.alpha.item()
            levels = module.quantizer.get_quantization_levels().cpu().numpy().tolist()
            print(f"  {name}: alpha={alpha:.4f}, levels={[f'{l:.3f}' for l in levels]}")
    
    # Save final model
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f'{checkpoint_dir}/mnist_qat_lsq_only_{timestamp}.pth'
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'best_acc': best_acc,
        'seed': args.seed,
        'num_bits': args.num_bits,
        'method': 'lsq_only_no_clipping',
    }, checkpoint_path)
    
    writer.close()
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("="*70)
    print()
    print("Compare with LSQ + BR:")
    print("  tensorboard --logdir=runs")
    print("  Look at activations/relu* histograms")
    print("  LSQ-only should show more spread, LSQ+BR should show sharper peaks")


if __name__ == '__main__':
    main()

