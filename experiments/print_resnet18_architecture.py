#!/usr/bin/env python3
"""
Print ResNet18 Architecture for CIFAR-10

This script loads ResNet18 (adapted for CIFAR-10) and prints:
1. Full model architecture
2. All ReLU layer locations
3. Layer count statistics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ClippedReLU(nn.Module):
    """ReLU with configurable clipping value."""
    def __init__(self, clip_value=6.0):
        super().__init__()
        self.clip_value = clip_value
    
    def forward(self, x):
        return torch.clamp(x, 0.0, self.clip_value)
    
    def __repr__(self):
        return f'ClippedReLU(clip_value={self.clip_value})'


def replace_relu_with_clipped(model, clip_value=6.0):
    """Replace all ReLU layers with ClippedReLU."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, ClippedReLU(clip_value=clip_value))
        else:
            replace_relu_with_clipped(module, clip_value)
    return model


def get_resnet18_cifar10(num_classes=10, pretrained=False, clip_value=None):
    """Get ResNet18 adapted for CIFAR-10."""
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    
    model = resnet18(weights=weights)
    
    # CIFAR-10 Adaptations
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Optionally replace ReLU with ClippedReLU
    if clip_value is not None:
        model = replace_relu_with_clipped(model, clip_value=clip_value)
    
    return model


def main():
    print("="*80)
    print("ResNet18 Architecture for CIFAR-10")
    print("="*80)
    print()
    
    # Create model (from scratch, no pretrained)
    model = get_resnet18_cifar10(num_classes=10, pretrained=False, clip_value=None)
    
    # Print full architecture
    print("="*80)
    print("FULL MODEL ARCHITECTURE:")
    print("="*80)
    print(model)
    print("="*80)
    print()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Count layer types
    conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    relu_count = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    
    print("="*80)
    print("LAYER STATISTICS:")
    print("="*80)
    print(f"Conv2d layers: {conv_count}")
    print(f"BatchNorm2d layers: {bn_count}")
    print(f"ReLU layers: {relu_count}")
    print(f"Linear layers: {linear_count}")
    print("="*80)
    print()
    
    # Print all ReLU layer locations
    print("="*80)
    print("ReLU LAYER LOCATIONS:")
    print("="*80)
    relu_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            relu_idx += 1
            print(f"  {relu_idx:2d}. {name}")
    print("="*80)
    print()
    
    # Print layer structure (high-level)
    print("="*80)
    print("HIGH-LEVEL STRUCTURE:")
    print("="*80)
    print("ResNet18 (CIFAR-10 adapted):")
    print("  - conv1: Conv2d(3, 64, 3x3, stride=1) [modified from 7x7, stride=2]")
    print("  - bn1: BatchNorm2d(64)")
    print("  - relu: ReLU")
    print("  - maxpool: Identity [removed for CIFAR-10]")
    print("  - layer1: 2 BasicBlocks (64 channels)")
    print("  - layer2: 2 BasicBlocks (128 channels)")
    print("  - layer3: 2 BasicBlocks (256 channels)")
    print("  - layer4: 2 BasicBlocks (512 channels)")
    print("  - avgpool: AdaptiveAvgPool2d")
    print("  - fc: Linear(512, 10) [modified from 1000 classes]")
    print("="*80)
    print()
    
    # Print BasicBlock structure
    print("="*80)
    print("BasicBlock STRUCTURE (each block):")
    print("="*80)
    print("BasicBlock:")
    print("  - conv1: Conv2d(in, out, 3x3)")
    print("  - bn1: BatchNorm2d(out)")
    print("  - relu: ReLU [nn.ReLU module - counted]")
    print("  - conv2: Conv2d(out, out, 3x3)")
    print("  - bn2: BatchNorm2d(out)")
    print("  - [residual connection: out += identity]")
    print("  - F.relu(out) [functional ReLU - NOT counted as module]")
    print()
    print("Note: Each BasicBlock has 1 nn.ReLU module.")
    print("      The second ReLU (after residual) is functional (F.relu).")
    print("="*80)
    print()
    
    # Show CIFAR-10 modifications
    print("="*80)
    print("CIFAR-10 ADAPTATIONS:")
    print("="*80)
    print("1. conv1: Changed from 7x7 stride=2 to 3x3 stride=1")
    print("   Reason: Preserve spatial resolution for 32x32 images")
    print()
    print("2. maxpool: Replaced with Identity (removed)")
    print("   Reason: Too aggressive downsampling for small images")
    print()
    print("3. fc: Changed from 1000 to 10 classes")
    print("   Reason: CIFAR-10 has 10 classes")
    print("="*80)


if __name__ == '__main__':
    main()

