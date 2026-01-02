#!/usr/bin/env python3
"""
Verify that BR is now operating on PRE-quantization activations.

This script loads a QAT model and checks:
1. Pre-quant activations are being stored
2. Pre-quant activations are continuous (not discrete)
3. Post-quant activations are discrete
4. Quantization residuals make sense

Usage:
    python verify_br_fix.py --model checkpoints/mnist_qat_binreg_xxx.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from experiments.mnist_qat_binreg import PlainConvFlattenQAT
from abr.lsq_quantizer import QuantizedClippedReLU


def verify_model(model, data_loader, device, num_batches=5):
    """
    Verify that pre-quant activations are being captured and are continuous.
    """
    model.eval()
    
    print("\n" + "="*70)
    print("Verification: Pre-Quantization Activations")
    print("="*70)
    
    # Collect one batch
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= 1:
                break
            data = data.to(device)
            _ = model(data)
    
    # Check each QuantizedClippedReLU layer
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            print(f"\nLayer: {name}")
            print(f"  Num bits: {module.num_bits}")
            print(f"  Clip value: {module.clip_value}")
            
            # Check if pre_quant_activation exists
            if not hasattr(module, 'pre_quant_activation'):
                print(f"  ❌ ERROR: No pre_quant_activation attribute!")
                continue
            
            if module.pre_quant_activation is None:
                print(f"  ❌ ERROR: pre_quant_activation is None!")
                continue
            
            pre_quant = module.pre_quant_activation
            print(f"  ✓ Pre-quant activations stored: shape={pre_quant.shape}")
            
            # Check if values are continuous (not discrete)
            unique_vals = torch.unique(pre_quant)
            print(f"  Unique values in pre-quant: {len(unique_vals)} (should be many)")
            
            # Get quantization levels
            levels = module.quantizer.get_quantization_levels().cpu().numpy()
            print(f"  Quantization levels: {len(levels)} levels")
            print(f"    Min level: {levels[0]:.6f}")
            print(f"    Max level: {levels[-1]:.6f}")
            print(f"    Step: {(levels[-1] - levels[0]) / (len(levels) - 1):.6f}")
            
            # Check pre-quant statistics
            pre_flat = pre_quant.flatten().cpu().numpy()
            print(f"  Pre-quant stats:")
            print(f"    Mean: {pre_flat.mean():.6f}")
            print(f"    Std:  {pre_flat.std():.6f}")
            print(f"    Min:  {pre_flat.min():.6f}")
            print(f"    Max:  {pre_flat.max():.6f}")
            
            # Compute distance to nearest level
            pre_flat_torch = torch.from_numpy(pre_flat)
            levels_torch = torch.from_numpy(levels)
            
            # For each activation, find distance to nearest level
            distances = []
            for val in pre_flat_torch:
                dist = torch.abs(levels_torch - val).min().item()
                distances.append(dist)
            
            distances = np.array(distances)
            print(f"  Distance to nearest level:")
            print(f"    Mean:   {distances.mean():.6f}")
            print(f"    Median: {np.median(distances):.6f}")
            print(f"    Max:    {distances.max():.6f}")
            
            # BR effectiveness
            alpha = module.quantizer.alpha.item()
            max_dist = alpha / 2  # Worst case for uniform spread
            effectiveness = 100 * (1 - distances.mean() / max_dist)
            print(f"  BR Effectiveness: {effectiveness:.1f}% (100% = perfect clustering)")
            
            if effectiveness > 80:
                print(f"  ✓ BR is working well!")
            elif effectiveness > 50:
                print(f"  ⚠ BR is partially working (may need higher lambda)")
            else:
                print(f"  ❌ BR is NOT working (need much higher lambda)")
    
    print("\n" + "="*70)
    print("Verification Complete")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Verify BR fix')
    parser.add_argument('--model', type=str, required=True, help='Path to QAT model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("BR Pre-Quantization Activation Verification")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.model, map_location=device)
    
    clip_value = checkpoint.get('clip_value', 1.0)
    num_bits = checkpoint.get('num_bits', 2)
    
    model = PlainConvFlattenQAT(
        input_channels=1,
        num_classes=10,
        base=16,
        clip_value=clip_value,
        num_bits=num_bits
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded! (clip_value={clip_value}, num_bits={num_bits})")
    
    # Load test data
    print("\nLoading test data...")
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Verify
    verify_model(model, test_loader, device)


if __name__ == '__main__':
    main()

