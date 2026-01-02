#!/usr/bin/env python3
"""
Plot QAT Model Activation Histograms with Quantization Levels

Shows histograms with the exact learned quantization level values marked.
Verifies that Bin Regularization creates peaks at the correct positions.

Usage:
    python plot_qat_histograms.py \
        --qat-model checkpoints/mnist_qat_binreg_xxx.pth \
        --output-dir figures_qat/
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from experiments.mnist_qat_binreg import PlainConvFlattenQAT, QuantizedClippedReLU

# Set style
matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18
})


def collect_activations_and_levels(model, data_loader, device, num_batches=20):
    """
    Collect activations and extract quantization levels.
    
    Returns:
        activations: Dict[layer_name, tensor] - Concatenated activations
        quant_info: Dict[layer_name, dict] - Quantization levels and alpha
    """
    model.eval()
    activations = {}
    quant_info = {}
    
    # First, extract quantization information
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            alpha = module.quantizer.alpha.item()
            levels = module.quantizer.get_quantization_levels().cpu().numpy()
            quant_info[name] = {
                'alpha': alpha,
                'levels': levels,
                'num_bits': module.num_bits
            }
            activations[name] = []
    
    # Register hooks to collect activations
    handles = []
    
    def make_hook(name):
        def hook(module, input, output):
            activations[name].append(output.detach().cpu())
        return hook
    
    for name, module in model.named_modules():
        if name in activations:
            handle = module.register_forward_hook(make_hook(name))
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
    for name in activations.keys():
        activations[name] = torch.cat(activations[name], dim=0).flatten().numpy()
    
    return activations, quant_info


def plot_histogram_with_levels(ax, activations, quant_info, layer_name):
    """
    Plot histogram with quantization levels marked.
    
    Shows:
    - Histogram of activations
    - Vertical lines at quantization levels
    - Exact values labeled on each line
    """
    levels = quant_info['levels']
    alpha = quant_info['alpha']
    num_bits = quant_info['num_bits']
    
    # Plot histogram
    counts, bins, patches = ax.hist(
        activations,
        bins=100,
        density=True,
        alpha=0.7,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add vertical lines at quantization levels
    colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    for i, level in enumerate(levels):
        color = colors[i % len(colors)]
        ax.axvline(level, color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        # Add text label with exact value
        y_pos = ax.get_ylim()[1] * 0.85 - (i * ax.get_ylim()[1] * 0.08)
        ax.text(
            level, y_pos,
            f'{level:.3f}',
            rotation=0,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
            fontsize=10,
            fontweight='bold'
        )
    
    # Stats text
    stats_text = (
        f'α = {alpha:.4f}\n'
        f'{num_bits}-bit ({len(levels)} levels)\n'
        f'Mean: {activations.mean():.3f}\n'
        f'Std: {activations.std():.3f}\n'
        f'Range: [{activations.min():.3f}, {activations.max():.3f}]'
    )
    
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=9,
        family='monospace'
    )
    
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{layer_name} (QAT with Bin Regularization)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_yscale('log')  # Log scale to see all peaks clearly


def main():
    parser = argparse.ArgumentParser(description='Plot QAT histograms with quantization levels')
    parser.add_argument('--qat-model', type=str, required=True,
                        help='Path to QAT model checkpoint')
    parser.add_argument('--num-batches', type=int, default=20,
                        help='Number of batches to collect')
    parser.add_argument('--output-dir', type=str, default='figures_qat',
                        help='Output directory for figures')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("QAT Activation Histogram Plotting")
    print("="*70)
    print(f"QAT model: {args.qat_model}")
    print(f"Device: {device}")
    print()
    
    # Load QAT model
    print("Loading QAT model...")
    checkpoint = torch.load(args.qat_model, map_location=device)
    
    clip_value = checkpoint.get('clip_value', 1.0)
    num_bits = checkpoint.get('num_bits', 2)
    
    print(f"  Clip value: {clip_value}")
    print(f"  Num bits: {num_bits}")
    
    model = PlainConvFlattenQAT(
        input_channels=1,
        num_classes=10,
        base=16,
        clip_value=clip_value,
        num_bits=num_bits
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded")
    
    # Load test data
    print("\nLoading test data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])
    
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Collect activations and quantization info
    print(f"\nCollecting activations from {args.num_batches} batches...")
    activations, quant_info = collect_activations_and_levels(model, loader, device, args.num_batches)
    
    print(f"✅ Collected activations from {len(activations)} layers")
    
    # Print quantization levels
    print("\nLearned Quantization Levels:")
    print("-" * 70)
    for name in sorted(activations.keys()):
        info = quant_info[name]
        print(f"{name}:")
        print(f"  α = {info['alpha']:.4f}")
        print(f"  Levels: {[f'{l:.4f}' for l in info['levels']]}")
    print("-" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot each layer
    print("\nGenerating plots...")
    
    layer_names = sorted(activations.keys())
    n_layers = len(layer_names)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4*n_layers))
    
    if n_layers == 1:
        axes = [axes]
    
    for ax, name in zip(axes, layer_names):
        plot_histogram_with_levels(ax, activations[name], quant_info[name], name)
    
    plt.tight_layout()
    
    # Save figure
    output_png = output_dir / 'qat_histograms_with_levels.png'
    output_pdf = output_dir / 'qat_histograms_with_levels.pdf'
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    
    print(f"\n✅ Saved figures:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    
    # Also create individual plots for each layer
    print("\nGenerating individual layer plots...")
    for name in layer_names:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(10, 6))
        plot_histogram_with_levels(ax_single, activations[name], quant_info[name], name)
        
        output_png_single = output_dir / f'{name}_histogram.png'
        output_pdf_single = output_dir / f'{name}_histogram.pdf'
        
        fig_single.savefig(output_png_single, dpi=300, bbox_inches='tight')
        fig_single.savefig(output_pdf_single, bbox_inches='tight')
        plt.close(fig_single)
        
        print(f"   {name}: {output_png_single}")
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)
    print()
    print("The plots show:")
    print("  - Histogram of activation distributions")
    print("  - Vertical dashed lines at learned quantization levels")
    print("  - Exact numerical values labeled on each line")
    print("  - Statistics (α, mean, std, range)")
    print()
    print("If Bin Regularization worked correctly, you should see:")
    print("  ✓ Sharp peaks at the quantization level positions")
    print("  ✓ Very little mass between the peaks")
    print("  ✓ Histogram peaks aligned with the labeled values")
    print("="*70)


if __name__ == '__main__':
    main()

