#!/usr/bin/env python3
"""
Activation Distribution Visualizer for Single Checkpoint

Analyze and visualize activation distributions from any checkpoint (baseline or BR-trained).
Creates individual histogram plots per ReLU layer with quantization levels overlaid.

Usage:
    # Analyze baseline model with PTQ
    python experiments/compare_activations.py \
        --checkpoint results/mnist_relu1_sweep/checkpoints/mnist_resnet18_clip1.0_seed42_*.pth \
        --dataset mnist \
        --num-bits 2 \
        --calibration-percentile 99.9 \
        --output-dir plots/mnist_baseline_relu1_2bit/

    # Analyze BR-trained model (automatically extracts learned alphas)
    python experiments/compare_activations.py \
        --checkpoint results/mnist_relu1_br_sweep/checkpoints/qat_br_seed42_bits2_lambda1.0_*.pth \
        --dataset mnist \
        --num-bits 2 \
        --output-dir plots/mnist_br_relu1_2bit_lambda1/

    # CIFAR-10 example
    python experiments/compare_activations.py \
        --checkpoint results/relu_sweep/checkpoints/cifar10_resnet18_clipNone_seed42_*.pth \
        --dataset cifar10 \
        --num-bits 4 \
        --calibration-percentile 100.0 \
        --output-dir plots/cifar10_baseline_relu_4bit/
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from abr.lsq_quantizer import QuantizedClippedReLU
from abr.hooks import ActivationHookManager


# ============================================================================
# Data Loaders
# ============================================================================

def get_cifar10_loader(batch_size=128, train=False):
    """Get CIFAR-10 loader for calibration."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)


def get_mnist_loader(batch_size=128, train=False):
    """Get MNIST loader for calibration."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)


# ============================================================================
# Model Loading
# ============================================================================

def load_model_from_checkpoint(checkpoint_path, dataset='mnist', num_bits=2):
    """
    Load model from checkpoint. Auto-detects if it's baseline or BR-trained.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dataset: 'mnist' or 'cifar10'
        num_bits: Number of quantization bits (required for BR models)
    
    Returns:
        model: The loaded model
        is_br_model: Boolean indicating if it's a BR-trained model
        clip_value: The clip value used (or None for standard ReLU)
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Detect model type from checkpoint
    # Check both top-level keys and inside model_state_dict/state_dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    is_br_model = any('quantizer' in key for key in state_dict.keys())
    
    # Extract clip_value if available
    clip_value = checkpoint.get('clip_value', None)
    
    # Determine input channels
    in_channels = 1 if dataset == 'mnist' else 3
    
    if is_br_model:
        print("  Detected: BR-trained model (QAT)")
        # Load BR model
        if dataset == 'mnist':
            from experiments.mnist_resnet18_qat_binreg_aws import get_resnet18_mnist_qat
            model = get_resnet18_mnist_qat(
                num_classes=10,
                pretrained_imagenet=False,
                clip_value=clip_value,
                num_bits=num_bits,
                pretrained_baseline=None
            )
        else:  # cifar10
            from experiments.cifar10_resnet18_qat_binreg import get_resnet18_cifar10_qat
            model = get_resnet18_cifar10_qat(
                num_classes=10,
                pretrained_imagenet=False,
                clip_value=clip_value,
                num_bits=num_bits,
                pretrained_baseline=None
            )
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print(f"  Clip value: {clip_value}")
        
    else:
        print("  Detected: Baseline FP32 model")
        # Load baseline model
        if dataset == 'mnist':
            # Import ClippedReLU and ResNet18 builder
            try:
                from experiments.mnist_automated_ptq_sweep import ClippedReLU, get_resnet18_mnist
                model = get_resnet18_mnist(clip_value=clip_value, num_classes=10)
            except ImportError:
                # Fallback: define ClippedReLU here
                class ClippedReLU(nn.Module):
                    def __init__(self, clip_value=None):
                        super().__init__()
                        self.clip_value = clip_value
                    def forward(self, x):
                        if self.clip_value is not None:
                            return torch.clamp(x, 0, self.clip_value)
                        return F.relu(x)
                
                from torchvision.models import resnet18
                model = resnet18(num_classes=10)
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # Replace ReLUs if needed
                if clip_value is not None:
                    for name, module in model.named_modules():
                        if isinstance(module, nn.ReLU):
                            parent_name = '.'.join(name.split('.')[:-1])
                            child_name = name.split('.')[-1]
                            parent = model.get_submodule(parent_name) if parent_name else model
                            setattr(parent, child_name, ClippedReLU(clip_value))
        else:  # cifar10
            from experiments.cifar10_resnet18_baseline import get_resnet18_cifar10
            model = get_resnet18_cifar10(pretrained=False, clip_value=clip_value)
        
        # Load weights
        model.load_state_dict(state_dict)
        
        print(f"  Clip value: {clip_value}")
    
    model.eval()
    return model, is_br_model, clip_value


# ============================================================================
# Activation Collection
# ============================================================================

def collect_activations(model, loader, device, num_batches=10, max_samples_per_layer=100000, use_train_mode=False):
    """
    Collect activations from all ReLU layers.
    
    For baseline models: Collects post-activation values (output)
    For BR models: Collects pre-quantization continuous values (module.pre_quant_activation)
    
    Args:
        use_train_mode: If True, set model to train() mode (uses batch BN stats)
    
    Returns:
        activations: Dict of {layer_name: torch.Tensor}
        layer_modules: Dict of {layer_name: module} for extracting alphas later
    """
    if use_train_mode:
        model.train()
    else:
        model.eval()
    activations = {}
    sample_counts = {}
    layer_modules = {}
    
    def make_hook(name, module):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
                sample_counts[name] = 0
                layer_modules[name] = module
            
            # For QuantizedClippedReLU, use pre-quantization activations
            # For baseline ReLU/ClippedReLU, use output
            if isinstance(module, QuantizedClippedReLU) and hasattr(module, 'pre_quant_activation'):
                # Get continuous values BEFORE quantization
                acts_to_collect = module.pre_quant_activation.detach().cpu().flatten()
            else:
                # Get output (already post-activation, no quantization)
                acts_to_collect = output.detach().cpu().flatten()
            
            # Memory-efficient: only keep samples if under limit
            if sample_counts[name] < max_samples_per_layer:
                needed = max_samples_per_layer - sample_counts[name]
                
                if len(acts_to_collect) <= needed:
                    activations[name].append(acts_to_collect)
                    sample_counts[name] += len(acts_to_collect)
                else:
                    # Sample randomly
                    indices = torch.randperm(len(acts_to_collect))[:needed]
                    activations[name].append(acts_to_collect[indices])
                    sample_counts[name] += needed
        return hook
    
    # Register hooks on all ReLU-like layers
    # IMPORTANT: Skip child modules (like .quantizer) to avoid duplicates
    hooks = []
    layer_count = 0
    
    for name, module in model.named_modules():
        # Skip child modules like "layer1.0.relu.quantizer" - we only want "layer1.0.relu"
        if '.quantizer' in name or '.alpha' in name:
            continue
        
        # Check for standard ReLU, ReLU6, ClippedReLU, or QuantizedClippedReLU
        is_relu_like = (
            isinstance(module, (nn.ReLU, nn.ReLU6)) or
            isinstance(module, QuantizedClippedReLU) or
            (hasattr(module, 'clip_value') and isinstance(module, nn.Module) and not isinstance(module, QuantizedClippedReLU))
        )
        
        if is_relu_like:
            layer_name = name if name else f"relu{layer_count}"
            hooks.append(module.register_forward_hook(make_hook(layer_name, module)))
            layer_count += 1
    
    print(f"\nCollecting activations from {layer_count} ReLU layers...")
    print(f"  Memory limit: {max_samples_per_layer:,} samples per layer")
    print(f"  Calibration batches: {num_batches}")
    
    # Collect activations
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)
            _ = model(data)
            
            # Early exit if all layers have enough samples
            if all(count >= max_samples_per_layer for count in sample_counts.values()):
                print(f"  Early stop at batch {batch_idx+1}: all layers have enough samples")
                break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate collected activations
    for name in activations:
        activations[name] = torch.cat(activations[name])
        print(f"  {name}: {len(activations[name]):,} samples")
    
    return activations, layer_modules


# ============================================================================
# Quantization Level Computation
# ============================================================================

def compute_ptq_levels(activations, num_bits, percentile=100.0):
    """
    Compute PTQ quantization levels using percentile calibration.
    
    Returns:
        scale: The quantization scale
        levels: Array of quantization levels
    """
    # Compute max value using percentile
    if isinstance(activations, torch.Tensor):
        acts_np = activations.cpu().numpy()
    else:
        acts_np = activations
    
    max_val = float(np.percentile(np.abs(acts_np), percentile))
    
    if max_val == 0:
        max_val = 1e-6
    
    # Compute scale
    qmax = 2 ** num_bits - 1
    scale = max_val / qmax
    
    # Compute levels
    levels = np.arange(0, qmax + 1) * scale
    
    return scale, levels


def compute_quantization_mse(activations, scale, num_bits):
    """
    Compute MSE between original and quantized activations.
    """
    if isinstance(activations, torch.Tensor):
        acts = activations.cpu().numpy()
    else:
        acts = activations
    
    # Quantize
    qmax = 2 ** num_bits - 1
    quantized = np.round(np.clip(acts / scale, 0, qmax)) * scale
    
    # MSE
    mse = np.mean((acts - quantized) ** 2)
    return mse


def extract_br_levels(module, num_bits):
    """
    Extract learned quantization levels from a BR-trained QuantizedClippedReLU module.
    
    Returns:
        alpha: The learned alpha (scale)
        levels: Array of quantization levels
    """
    if not isinstance(module, QuantizedClippedReLU):
        raise ValueError("Module is not a QuantizedClippedReLU")
    
    alpha = module.quantizer.alpha.item()
    qmax = 2 ** num_bits - 1
    levels = np.arange(0, qmax + 1) * alpha
    
    return alpha, levels


# ============================================================================
# Plotting
# ============================================================================

def plot_activation_histogram(activations, levels, scale, layer_name, output_path, 
                               model_type="Baseline PTQ", num_bits=2, clip_value=None, mse=None):
    """
    Plot activation histogram with quantization levels overlaid.
    Creates 3 versions: zoomed (auto), log-scale, and full-range.
    """
    # Convert to numpy
    if isinstance(activations, torch.Tensor):
        acts = activations.cpu().numpy()
    else:
        acts = activations
    
    # Get base path without extension
    base_path = output_path.rsplit('.', 1)[0]
    ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
    
    # Determine full range x-limit
    if clip_value is not None:
        full_range_limit = clip_value * 1.05  # Clip value + 5%
    else:
        full_range_limit = 15.0  # Standard ReLU cap at 15
    
    # Common info text
    info_text_base = f'Quantization Levels (red lines)\n'
    info_text_base += f'#Bits: {num_bits}\n'
    info_text_base += f'#Levels: {len(levels)}\n'
    info_text_base += f'Min/Max: {acts.min():.4f} / {acts.max():.4f}\n'
    if mse is not None:
        info_text_base += f'MSE: {mse:.6f}'
    
    # ========== 1. ZOOMED LINEAR SCALE PLOT ==========
    max_level = levels[-1]
    zoom_limit = max(max_level * 1.5, float(np.percentile(acts, 95)))
    
    info_text_zoom = info_text_base + f'\nX-axis: zoomed to {zoom_limit:.3f}'
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Title
    title = f'{layer_name} - {model_type} ({num_bits}-bit)'
    if clip_value is not None:
        title += f' [Clip={clip_value}]'
    title += '\nPost-Activation, Pre-Quantization (Zoomed)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot histogram
    ax.hist(acts, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
    
    # Overlay quantization levels
    for level in levels:
        ax.axvline(level, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Labels and grid
    ax.set_xlabel('Activation Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Scale: {scale:.6f} | Samples: {len(acts):,}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, zoom_limit])
    
    # Add info box
    ax.text(0.98, 0.97, info_text_zoom,
            transform=ax.transAxes, 
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9,
            family='monospace')
    
    fig.tight_layout()
    zoomed_path = f"{base_path}_zoomed.{ext}"
    fig.savefig(zoomed_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved (zoomed):     {zoomed_path}")
    
    # ========== 2. LOG SCALE PLOT (ZOOMED) ==========
    info_text_log = info_text_base + f'\nX-axis: zoomed to {zoom_limit:.3f}'
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Title
    title = f'{layer_name} - {model_type} ({num_bits}-bit)'
    if clip_value is not None:
        title += f' [Clip={clip_value}]'
    title += '\nPost-Activation, Pre-Quantization (Log Scale - Zoomed)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot histogram with log scale
    counts, bins, patches = ax.hist(acts, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
    ax.set_yscale('log')
    
    # Overlay quantization levels
    y_max = counts.max() if len(counts) > 0 else 1
    for level in levels:
        ax.axvline(level, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Labels and grid
    ax.set_xlabel('Activation Value', fontsize=11)
    ax.set_ylabel('Count (log scale)', fontsize=11)
    ax.set_title(f'Scale: {scale:.6f} | Samples: {len(acts):,}', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0, zoom_limit])
    
    # Set y-axis range to avoid issues with zero counts
    if len(counts) > 0:
        min_nonzero = counts[counts > 0].min() if np.any(counts > 0) else 0.5
        ax.set_ylim([max(0.5, min_nonzero * 0.5), y_max * 2])
    
    # Add info box
    ax.text(0.98, 0.97, info_text_log,
            transform=ax.transAxes, 
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9,
            family='monospace')
    
    fig.tight_layout()
    log_path = f"{base_path}_log.{ext}"
    fig.savefig(log_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved (log):        {log_path}")
    
    # ========== 3. FULL RANGE PLOT ==========
    info_text_full = info_text_base + f'\nX-axis: full range [0, {full_range_limit:.1f}]'
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Title
    title = f'{layer_name} - {model_type} ({num_bits}-bit)'
    if clip_value is not None:
        title += f' [Clip={clip_value}]'
    title += '\nPost-Activation, Pre-Quantization (Full Range)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot histogram
    ax.hist(acts, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
    
    # Overlay quantization levels
    for level in levels:
        ax.axvline(level, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Labels and grid
    ax.set_xlabel('Activation Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Scale: {scale:.6f} | Samples: {len(acts):,}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, full_range_limit])
    
    # Add info box
    ax.text(0.98, 0.97, info_text_full,
            transform=ax.transAxes, 
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9,
            family='monospace')
    
    fig.tight_layout()
    full_path = f"{base_path}_fullrange.{ext}"
    fig.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved (full range): {full_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize activation distributions from a single checkpoint')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (supports wildcards)')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'],
                        help='Dataset: mnist or cifar10')
    parser.add_argument('--num-bits', type=int, required=True,
                        help='Number of quantization bits')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save plots')
    
    # Optional arguments
    parser.add_argument('--calibration-percentile', type=float, default=100.0,
                        help='Percentile for PTQ calibration (default: 100.0)')
    parser.add_argument('--calibration-batches', type=int, default=10,
                        help='Number of batches for calibration (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for data loading (default: 128)')
    parser.add_argument('--max-samples', type=int, default=100000,
                        help='Max samples per layer (default: 100000)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use (default: 0)')
    parser.add_argument('--use-train-data', action='store_true',
                        help='Use training data instead of test data (matches BR training distribution)')
    parser.add_argument('--train-mode', action='store_true',
                        help='Set model to train mode (uses batch BN stats instead of running stats)')
    parser.add_argument('--plot-top-n', type=int, default=None,
                        help='Only plot top N layers with lowest MSE (default: plot all)')
    
    args = parser.parse_args()
    
    # Resolve checkpoint path (handle wildcards)
    checkpoint_paths = glob(args.checkpoint)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint found matching: {args.checkpoint}")
    checkpoint_path = checkpoint_paths[0]  # Use first match
    if len(checkpoint_paths) > 1:
        print(f"Warning: Multiple checkpoints found. Using: {checkpoint_path}")
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, is_br_model, clip_value = load_model_from_checkpoint(
        checkpoint_path, args.dataset, num_bits=args.num_bits
    )
    model = model.to(device)
    
    # Get data loader
    if args.dataset == 'mnist':
        loader = get_mnist_loader(args.batch_size, train=args.use_train_data)
    else:
        loader = get_cifar10_loader(args.batch_size, train=args.use_train_data)
    
    if args.use_train_data:
        print(f"Using TRAINING data for activation collection")
    else:
        print(f"Using TEST data for activation collection")
    
    # Collect activations
    activations, layer_modules = collect_activations(
        model, loader, device, 
        num_batches=args.calibration_batches,
        max_samples_per_layer=args.max_samples,
        use_train_mode=args.train_mode
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each layer - compute MSE first
    print(f"\nComputing MSE for all layers...")
    model_type = "QAT-BR" if is_br_model else "Baseline PTQ"
    layer_mse = {}
    
    for layer_name, acts in activations.items():
        # Get quantization levels
        if is_br_model and layer_name in layer_modules:
            module = layer_modules[layer_name]
            if isinstance(module, QuantizedClippedReLU):
                # Extract learned alpha from BR model
                scale, levels = extract_br_levels(module, args.num_bits)
            else:
                # Fallback to PTQ for non-quantized layers
                scale, levels = compute_ptq_levels(acts, args.num_bits, args.calibration_percentile)
        else:
            # PTQ calibration
            scale, levels = compute_ptq_levels(acts, args.num_bits, args.calibration_percentile)
        
        # Compute MSE
        mse = compute_quantization_mse(acts, scale, args.num_bits)
        layer_mse[layer_name] = (mse, scale, levels, acts)
        print(f"  {layer_name}: scale={scale:.6f}, MSE={mse:.6f}")
    
    # Sort by MSE (best first)
    sorted_layers = sorted(layer_mse.items(), key=lambda x: x[1][0])
    
    print(f"\n{'='*80}")
    print(f"Top 5 layers with BEST quantization (lowest MSE):")
    for i, (layer_name, (mse, scale, _, _)) in enumerate(sorted_layers[:5]):
        print(f"  {i+1}. {layer_name}: MSE={mse:.6f}, scale={scale:.6f}")
    
    print(f"\nBottom 3 layers with WORST quantization (highest MSE):")
    for i, (layer_name, (mse, scale, _, _)) in enumerate(sorted_layers[-3:]):
        print(f"  {len(sorted_layers)-2+i}. {layer_name}: MSE={mse:.6f}, scale={scale:.6f}")
    print(f"{'='*80}")
    
    # Generate plots
    print(f"\nGenerating plots...")
    
    # Determine which layers to plot
    if args.plot_top_n is not None and args.plot_top_n > 0:
        layers_to_plot = sorted_layers[:args.plot_top_n]
        print(f"  Plotting TOP {args.plot_top_n} layers with lowest MSE")
    else:
        layers_to_plot = sorted_layers
        print(f"  Plotting ALL {len(sorted_layers)} layers")
    
    for layer_name, (mse, scale, levels, acts) in layers_to_plot:
        output_path = os.path.join(args.output_dir, f'{layer_name}_histogram.png')
        plot_activation_histogram(
            acts, levels, scale, layer_name, output_path,
            model_type=model_type,
            num_bits=args.num_bits,
            clip_value=clip_value,
            mse=mse
        )
    
    print(f"\n✓ Plots saved to: {args.output_dir}")
    print(f"  Plotted: {len(layers_to_plot)} / {len(activations)} layers")
    print(f"  Average MSE: {np.mean([mse for mse, _, _, _ in layer_mse.values()]):.6f}")
    print(f"  Best MSE: {sorted_layers[0][1][0]:.6f} ({sorted_layers[0][0]})")
    print(f"  Worst MSE: {sorted_layers[-1][1][0]:.6f} ({sorted_layers[-1][0]})")


if __name__ == '__main__':
    main()
