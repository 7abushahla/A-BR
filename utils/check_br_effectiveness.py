#!/usr/bin/env python3
"""
Check if Bin Regularization is actually working.

Measures the PRE-quantization activations (before rounding) and checks:
1. How close they are to the nearest quantization level
2. The distribution within each bin
3. Whether BR is creating sharp peaks vs. spread distributions

Usage:
    python check_br_effectiveness.py \
        --qat-model checkpoints/mnist_qat_binreg_xxx.pth \
        --num-bits 4
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from experiments.mnist_qat_binreg import PlainConvFlattenQAT, QuantizedClippedReLU
from abr.lsq_quantizer import LSQ_ActivationQuantizer


def collect_pre_quantization_activations(model, data_loader, device, num_batches=50):
    """
    Collect activations BEFORE quantization (before round_pass).
    
    We hook into the ReLU output (before it goes into the quantizer).
    """
    model.eval()
    pre_quant_activations = {}
    quant_info = {}
    
    # First, extract quantization information
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            alpha = module.quantizer.alpha.item()
            Qp = module.quantizer.Qp
            levels = module.quantizer.get_quantization_levels().cpu().numpy()
            quant_info[name] = {
                'alpha': alpha,
                'Qp': Qp,
                'levels': levels,
                'num_bits': module.num_bits
            }
            pre_quant_activations[name] = []
    
    # Register hooks to collect PRE-quantization activations
    handles = []
    
    def make_pre_quant_hook(name, module):
        """Hook to capture activations after ReLU but BEFORE quantization."""
        def hook(m, input, output):
            # output here is after torch.clamp(F.relu(x), max=clip_value)
            # but BEFORE the LSQ quantizer's round_pass
            pre_quant_activations[name].append(output.detach().cpu())
        return hook
    
    # Hook at the QuantizedClippedReLU level to get pre-quantized values
    # We need to modify the hook to capture BEFORE quantizer is applied
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            # We'll use a custom pre-hook on the quantizer itself
            def make_quantizer_pre_hook(layer_name):
                def pre_hook(quantizer_module, input):
                    # input[0] is the tensor going INTO the quantizer
                    # This is the clipped ReLU output, before quantization
                    pre_quant_activations[layer_name].append(input[0].detach().cpu())
                return pre_hook
            
            handle = module.quantizer.register_forward_pre_hook(make_quantizer_pre_hook(name))
            handles.append(handle)
    
    # Collect activations
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)
            _ = model(data)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Concatenate batches
    for name in pre_quant_activations.keys():
        if len(pre_quant_activations[name]) > 0:
            pre_quant_activations[name] = torch.cat(pre_quant_activations[name], dim=0).flatten().numpy()
        else:
            pre_quant_activations[name] = np.array([])
    
    return pre_quant_activations, quant_info


def analyze_br_effectiveness(activations, levels, alpha, layer_name):
    """
    Analyze how well BR is working by measuring:
    1. Distance from each activation to nearest level
    2. Variance within each bin
    3. Percentage of activations near levels (within epsilon)
    """
    print(f"\n{'='*70}")
    print(f"Layer: {layer_name}")
    print(f"{'='*70}")
    print(f"Alpha: {alpha:.6f}")
    print(f"Num levels: {len(levels)}")
    print(f"Num activations: {len(activations)}")
    print(f"Range: [{activations.min():.4f}, {activations.max():.4f}]")
    
    # 1. Distance to nearest level
    distances = []
    for act in activations:
        # Find nearest level
        dist = np.abs(levels - act).min()
        distances.append(dist)
    
    distances = np.array(distances)
    
    print(f"\nDistance to nearest level:")
    print(f"  Mean:   {distances.mean():.6f}")
    print(f"  Median: {np.median(distances):.6f}")
    print(f"  Std:    {distances.std():.6f}")
    print(f"  Max:    {distances.max():.6f}")
    
    # 2. Percentage within epsilon of levels
    for eps_mult in [0.01, 0.05, 0.1]:
        eps = eps_mult * alpha
        within_eps = (distances < eps).sum()
        pct = 100.0 * within_eps / len(distances)
        print(f"  Within {eps_mult}α ({eps:.6f}): {within_eps}/{len(distances)} ({pct:.2f}%)")
    
    # 3. Bin-wise statistics
    print(f"\nPer-bin analysis:")
    bin_edges = np.concatenate([levels, [levels[-1] + alpha]])  # Add upper edge
    bin_indices = np.digitize(activations, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(levels) - 1)
    
    total_mse = 0
    total_var = 0
    non_empty_bins = 0
    
    for i, level in enumerate(levels):
        bin_vals = activations[bin_indices == i]
        if len(bin_vals) > 0:
            bin_mean = bin_vals.mean()
            bin_var = bin_vals.var() if len(bin_vals) > 1 else 0
            bin_mse = ((bin_vals - level) ** 2).mean()
            
            total_mse += bin_mse * len(bin_vals)
            total_var += bin_var * len(bin_vals)
            non_empty_bins += 1
            
            if i < 5 or i >= len(levels) - 5:  # Print first/last 5 bins
                print(f"  Bin {i:2d} (level={level:.4f}): "
                      f"n={len(bin_vals):5d}, "
                      f"mean={bin_mean:.4f}, "
                      f"var={bin_var:.6f}, "
                      f"mse={bin_mse:.6f}")
    
    avg_mse = total_mse / len(activations)
    avg_var = total_var / len(activations)
    
    print(f"\nOverall bin statistics:")
    print(f"  Non-empty bins: {non_empty_bins}/{len(levels)}")
    print(f"  Avg MSE:  {avg_mse:.6f}")
    print(f"  Avg Var:  {avg_var:.6f}")
    
    # 4. BR effectiveness score
    # Perfect BR: all activations exactly at levels → distance mean = 0
    # Poor BR: activations spread uniformly → distance mean ≈ α/2
    max_dist = alpha / 2  # Worst case: uniform spread
    effectiveness = 100 * (1 - distances.mean() / max_dist)
    print(f"\nBR Effectiveness Score: {effectiveness:.2f}% (100% = perfect clustering)")
    
    return {
        'mean_distance': distances.mean(),
        'median_distance': np.median(distances),
        'avg_mse': avg_mse,
        'avg_var': avg_var,
        'effectiveness': effectiveness
    }


def main():
    parser = argparse.ArgumentParser(description='Check BR effectiveness')
    parser.add_argument('--qat-model', type=str, required=True)
    parser.add_argument('--num-bits', type=int, default=4)
    parser.add_argument('--num-batches', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("Bin Regularization Effectiveness Analysis")
    print("="*70)
    print(f"Model: {args.qat_model}")
    print(f"Device: {device}")
    print(f"Num bits: {args.num_bits}")
    print(f"="*70)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.qat_model, map_location=device)
    
    clip_value = checkpoint.get('clip_value', 1.0)
    model = PlainConvFlattenQAT(
        input_channels=1,
        num_classes=10,
        base=16,
        clip_value=clip_value,
        num_bits=args.num_bits
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded! (clip_value={clip_value})")
    
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
        shuffle=False,
        num_workers=2
    )
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Collect pre-quantization activations
    print(f"\nCollecting pre-quantization activations ({args.num_batches} batches)...")
    activations, quant_info = collect_pre_quantization_activations(
        model, test_loader, device, num_batches=args.num_batches
    )
    print("Done!")
    
    # Analyze each layer
    results = {}
    for layer_name in sorted(activations.keys()):
        if len(activations[layer_name]) == 0:
            print(f"\nSkipping {layer_name} (no activations collected)")
            continue
        
        results[layer_name] = analyze_br_effectiveness(
            activations[layer_name],
            quant_info[layer_name]['levels'],
            quant_info[layer_name]['alpha'],
            layer_name
        )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: BR Effectiveness Across Layers")
    print("="*70)
    print(f"{'Layer':<10} {'Mean Dist':>12} {'Avg MSE':>12} {'Avg Var':>12} {'Score':>10}")
    print("-"*70)
    for layer_name in sorted(results.keys()):
        r = results[layer_name]
        print(f"{layer_name:<10} {r['mean_distance']:>12.6f} {r['avg_mse']:>12.6f} "
              f"{r['avg_var']:>12.6f} {r['effectiveness']:>9.2f}%")
    
    print("="*70)
    print("\nInterpretation:")
    print("  - Mean Dist: Lower is better (0 = perfect clustering)")
    print("  - Avg MSE/Var: Lower is better (0 = no spread within bins)")
    print("  - Score: 100% = perfect, <50% = BR not effective")
    print("="*70)


if __name__ == '__main__':
    main()

