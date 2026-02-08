#!/usr/bin/env python3
"""
Comprehensive Quantization Evaluation: Baseline PTQ vs QAT-BR

This script performs a fair comparison between:
1. Baseline (FP32 trained) + PTQ with calibration
2. QAT-BR (trained with BR) using learned scales

The goal is to show that BR reduces quantization error and maintains accuracy.

Usage:
    # Evaluate both baseline PTQ and QAT-BR
    python experiments/evaluate_quantization.py \
        --baseline-model checkpoints/mnist_baseline.pth \
        --qat-model checkpoints/mnist_qat_binreg.pth \
        --num-bits 2 \
        --output-dir results/quantization_comparison

    # Baseline PTQ only
    python experiments/evaluate_quantization.py \
        --baseline-model checkpoints/mnist_baseline.pth \
        --num-bits 2 \
        --mode ptq

    # QAT-BR evaluation only
    python experiments/evaluate_quantization.py \
        --qat-model checkpoints/mnist_qat_binreg.pth \
        --num-bits 2 \
        --mode qat
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from abr.lsq_quantizer import QuantizedClippedReLU
from abr.hooks import ActivationHookManager

# Import model architectures from existing scripts
from experiments.mnist_baseline import PlainConvFlatten
from experiments.mnist_qat_binreg import PlainConvFlattenQAT


# ============================================================================
# PTQ Calibration & Quantization
# ============================================================================

class ActivationQuantizer:
    """
    Simple uniform quantizer for activations with calibration.
    
    This is used for PTQ on baseline models.
    """
    def __init__(self, num_bits: int):
        self.num_bits = num_bits
        self.scale = None
        self.qmin = 0
        self.qmax = 2 ** num_bits - 1
        self.is_calibrated = False
    
    def calibrate(self, activations: torch.Tensor, percentile=99.0):
        """
        Calibrate scale based on percentile of activation values.
        This is much better than max for handling outliers!
        """
        # Use numpy percentile (handles large tensors better)
        acts_flat = activations.flatten().cpu().numpy()
        max_val = float(np.percentile(np.abs(acts_flat), percentile))
        
        if max_val == 0:
            self.scale = 1.0
        else:
            self.scale = max_val / self.qmax
        self.is_calibrated = True
        actual_max = float(np.max(np.abs(acts_flat)))
        print(f"  Calibrated: {percentile}th percentile={max_val:.4f}, actual_max={actual_max:.4f}")
        print(f"             scale={self.scale:.6f}, levels=[0, {self.scale:.4f}, ..., {self.qmax*self.scale:.4f}]")

    def calibrate_mse_search(
        self,
        activations: torch.Tensor,
        candidate_percentiles=(50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0, 99.5, 99.9),
        sample_size: int = 200_000,
    ):
        """
        Calibrate by searching for the percentile/range that minimizes activation reconstruction MSE.

        This is a stronger PTQ baseline than naive max/percentile, especially for low bits.
        """
        acts = activations.flatten()
        if acts.numel() == 0:
            self.scale = 1.0
            self.is_calibrated = True
            return

        # Sample for speed (acts can be huge)
        if acts.numel() > sample_size:
            idx = torch.randint(0, acts.numel(), (sample_size,), device=acts.device)
            acts = acts[idx]

        # Move to CPU numpy once
        acts_np = acts.detach().cpu().numpy()
        abs_np = np.abs(acts_np)
        actual_max = float(abs_np.max()) if abs_np.size > 0 else 0.0

        best = {
            'percentile': None,
            'max_val': None,
            'scale': None,
            'mse': float('inf'),
            'pct_qmax': None,
            'pct_qmin': None,
        }

        for p in candidate_percentiles:
            max_val = float(np.percentile(abs_np, p))
            if max_val <= 0:
                continue
            scale = max_val / self.qmax

            # Simulate quant-dequant on the sampled activations
            x = torch.from_numpy(acts_np).to(dtype=torch.float32)
            x_int = torch.round(x / scale)
            x_int = torch.clamp(x_int, self.qmin, self.qmax)
            x_q = x_int * scale
            mse = float(((x - x_q) ** 2).mean().item())

            pct_qmin = float((x_int == self.qmin).float().mean().item() * 100.0)
            pct_qmax = float((x_int == self.qmax).float().mean().item() * 100.0)

            if mse < best['mse']:
                best.update(
                    percentile=float(p),
                    max_val=float(max_val),
                    scale=float(scale),
                    mse=float(mse),
                    pct_qmin=pct_qmin,
                    pct_qmax=pct_qmax,
                )

        if best['scale'] is None:
            # Fallback
            self.scale = 1.0
            self.is_calibrated = True
            return

        self.scale = best['scale']
        self.is_calibrated = True
        print(
            f"  Calibrated (MSE-search): best_p={best['percentile']:.1f} "
            f"max_val={best['max_val']:.4f} actual_max={actual_max:.4f} "
            f"scale={self.scale:.6f} mse(sample)={best['mse']:.6f} "
            f"%qmin={best['pct_qmin']:.1f}% %qmax={best['pct_qmax']:.1f}% "
            f"levels=[0, {self.scale:.4f}, ..., {self.qmax*self.scale:.4f}]"
        )
    
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (simulates quantization noise)."""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before quantizing!")
        # Match LSQ exactly: clamp BEFORE round
        x_int = torch.round(torch.clamp(x / self.scale, self.qmin, self.qmax))
        return x_int * self.scale

    def quantize_int(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to integer codes in [qmin, qmax]."""
        if not self.is_calibrated:
            raise ValueError("Must calibrate before quantizing!")
        # Match LSQ exactly: clamp BEFORE round
        x_int = torch.round(torch.clamp(x / self.scale, self.qmin, self.qmax))
        return x_int
    
    def get_levels(self):
        """Get quantization levels for visualization."""
        if not self.is_calibrated:
            return None
        return torch.arange(self.qmin, self.qmax + 1) * self.scale


def collect_activations_for_calibration(model, loader, device, num_batches=50):
    """
    Collect activations from ReLU layers for PTQ calibration.
    
    Returns dict of {layer_name: concatenated_activations}
    """
    model.eval()
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            # Collect output activations
            activations[name].append(output.detach().cpu())
        return hook
    
    # Import ClippedReLU from baseline script
    from experiments.mnist_baseline import ClippedReLU
    
    # Register hooks on all ReLU/clamp operations
    hooks = []
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.ReLU6, ClippedReLU)):
            layer_name = f"relu{layer_count + 1}"
            hooks.append(module.register_forward_hook(make_hook(layer_name)))
            layer_count += 1
    
    print(f"Collecting activations from {layer_count} ReLU layers for calibration...")
    
    # Collect activations
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)
            _ = model(data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate activations across batches
    for name in activations:
        activations[name] = torch.cat([a.flatten() for a in activations[name]])
    
    print(f"Collected activations for {len(activations)} layers")
    return activations


def calibrate_ptq_quantizers(activations_dict, num_bits, percentile=99.0, method: str = "percentile"):
    """Create and calibrate PTQ quantizers for each layer."""
    quantizers = {}
    if method == "percentile":
        print(f"\nCalibrating PTQ quantizers (using {percentile}th percentile):")
    else:
        print(f"\nCalibrating PTQ quantizers (method={method}):")
    for layer_name, acts in activations_dict.items():
        print(f"  {layer_name}:")
        q = ActivationQuantizer(num_bits)
        if method == "percentile":
            q.calibrate(acts, percentile=percentile)
        elif method == "mse_search":
            q.calibrate_mse_search(acts)
        else:
            raise ValueError(f"Unknown PTQ calibration method: {method}")
        quantizers[layer_name] = q
    return quantizers


def apply_ptq_quantization(model, quantizers, loader, device):
    """
    Apply PTQ quantization during inference by wrapping activation layers.
    
    Returns: accuracy, quantized_activations_dict, original_activations_dict
    """
    model.eval()
    correct = 0
    total = 0
    pred_hist = None
    
    quantized_acts_all = {name: [] for name in quantizers.keys()}
    original_acts_all = {name: [] for name in quantizers.keys()}
    
    # Import ClippedReLU from baseline script
    from experiments.mnist_baseline import ClippedReLU
    
    # Wrap each ReLU with a quantizer
    relu_modules = []
    original_forwards = []
    layer_idx = 0
    sanity = {
        'layers': {},
        'num_batches': 0,
    }
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.ReLU6, ClippedReLU)):
            layer_name = f"relu{layer_idx + 1}"
            if layer_name in quantizers:
                relu_modules.append((module, layer_name, quantizers[layer_name]))
                original_forwards.append(module.forward)
                
                # Create new forward that quantizes
                def make_quantized_forward(original_forward, layer_name, quantizer, acts_orig, acts_quant):
                    def quantized_forward(x):
                        # Apply original activation
                        output = original_forward(x)
                        # Store original
                        acts_orig[layer_name].append(output.detach().cpu())
                        # Quantize (and keep integer codes for sanity checks)
                        x_int = quantizer.quantize_int(output)
                        output_quant = x_int * quantizer.scale
                        # Store quantized
                        acts_quant[layer_name].append(output_quant.detach().cpu())
                        # Sanity: track saturation rates (0 and qmax) without storing huge tensors
                        if layer_name not in sanity['layers']:
                            sanity['layers'][layer_name] = {
                                'total_elems': 0,
                                'num_qmin': 0,
                                'num_qmax': 0,
                                'example_unique_codes': None,
                            }
                        layer_s = sanity['layers'][layer_name]
                        layer_s['total_elems'] += x_int.numel()
                        layer_s['num_qmin'] += (x_int == quantizer.qmin).sum().item()
                        layer_s['num_qmax'] += (x_int == quantizer.qmax).sum().item()
                        # Store an example of unique codes from the first time we see this layer
                        if layer_s['example_unique_codes'] is None:
                            # Limit cost: sample up to 200k elements
                            flat = x_int.flatten()
                            if flat.numel() > 200_000:
                                idx = torch.randint(0, flat.numel(), (200_000,), device=flat.device)
                                flat = flat[idx]
                            layer_s['example_unique_codes'] = torch.unique(flat).detach().cpu().tolist()
                        # Return quantized (this actually flows through network!)
                        return output_quant
                    return quantized_forward
                
                # Replace forward method
                module.forward = make_quantized_forward(
                    original_forwards[-1], layer_name, quantizers[layer_name],
                    original_acts_all, quantized_acts_all
                )
            layer_idx += 1
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
            total += len(target)
            sanity['num_batches'] += 1

            # Prediction histogram to detect collapse (e.g., always predicting one class)
            preds = pred.view(-1)
            if pred_hist is None:
                pred_hist = torch.zeros(10, dtype=torch.long)
            pred_hist += torch.bincount(preds.cpu(), minlength=10)
    
    # Restore original forward methods
    for (module, _, _), orig_forward in zip(relu_modules, original_forwards):
        module.forward = orig_forward

    # Print sanity summary: if PTQ is really applied, codes should be subset of {0,1,2,3}
    print("\nPTQ SANITY CHECK (activation quantization during inference):")
    for layer_name in sorted(sanity['layers'].keys()):
        s = sanity['layers'][layer_name]
        layer_total = max(1, s['total_elems'])  # DON'T overwrite 'total'!
        pct_qmin = 100.0 * s['num_qmin'] / layer_total
        pct_qmax = 100.0 * s['num_qmax'] / layer_total
        uniq = s['example_unique_codes']
        print(f"  {layer_name}: unique_codes(example)={uniq} | %qmin={pct_qmin:.2f}% | %qmax={pct_qmax:.2f}%")

    # Check for prediction collapse
    if pred_hist is not None:
        top_frac = (pred_hist.max().item() / max(1, pred_hist.sum().item())) * 100.0
        if top_frac > 50.0:  # Severe collapse
            print(f"\n⚠️  WARNING: Prediction collapse detected ({top_frac:.1f}% predict one class)")
    
    # Concatenate activations
    quantized_acts = {name: torch.cat([a.flatten() for a in acts]) 
                     for name, acts in quantized_acts_all.items()}
    original_acts = {name: torch.cat([a.flatten() for a in acts]) 
                    for name, acts in original_acts_all.items()}
    
    accuracy = 100. * correct / total
    print(f"\n✓ PTQ evaluation complete: {correct}/{total} correct = {accuracy:.2f}%")
    return accuracy, quantized_acts, original_acts


# ============================================================================
# QAT Evaluation (uses learned scales from training)
# ============================================================================

def evaluate_qat_model(model, loader, device):
    """
    Evaluate QAT model with its learned quantization scales.
    
    Returns: accuracy, quantized_activations_dict, original_activations_dict, scales_dict
    """
    model.eval()
    correct = 0
    total = 0
    
    # Get learned scales from model
    scales_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            scales_dict[name] = module.quantizer.alpha.item()
    
    print("\nLearned scales from QAT training:")
    for name, scale in scales_dict.items():
        num_bits = model.num_bits
        qmax = 2 ** num_bits - 1
        print(f"  {name}: alpha={scale:.6f}, levels=[0, {scale:.4f}, ..., {qmax*scale:.4f}]")
    
    # Use hook manager to collect activations
    hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],
        layer_names=list(scales_dict.keys()),
        exclude_first_last=False,
        detach_activations=True
    )
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
    
    # Get activations
    quantized_acts = hook_manager.get_activations()  # Post-quantization
    original_acts = hook_manager.get_pre_quant_activations()  # Pre-quantization
    
    # Flatten for analysis
    quantized_acts = {name: acts.flatten() for name, acts in quantized_acts.items()}
    original_acts = {name: acts.flatten() for name, acts in original_acts.items()}
    
    accuracy = 100. * correct / total
    return accuracy, quantized_acts, original_acts, scales_dict


# ============================================================================
# Metrics & Analysis
# ============================================================================

def compute_quantization_mse(original_acts, quantized_acts):
    """Compute MSE between original and quantized activations."""
    mse_dict = {}
    for layer_name in original_acts.keys():
        if layer_name in quantized_acts:
            orig = original_acts[layer_name]
            quant = quantized_acts[layer_name]
            mse = ((orig - quant) ** 2).mean().item()
            mse_dict[layer_name] = mse
    return mse_dict


def compute_clustering_effectiveness(original_acts, quantized_acts):
    """Compute how well activations cluster around discrete levels."""
    eff_dict = {}
    for layer_name in original_acts.keys():
        if layer_name in quantized_acts:
            orig = original_acts[layer_name]
            quant = quantized_acts[layer_name]
            
            # Mean distance between original and quantized
            mean_dist = (orig - quant).abs().mean().item()
            
            # Step size (distance between quantized values)
            unique_quant = torch.unique(quant)
            if len(unique_quant) > 1:
                step_size = (unique_quant.max() - unique_quant.min()).item() / (len(unique_quant) - 1)
            else:
                step_size = 1.0
            
            # Effectiveness: 1 - (mean_dist / (step_size/2))
            max_possible_dist = step_size / 2
            effectiveness = 100.0 * (1.0 - mean_dist / (max_possible_dist + 1e-12))
            effectiveness = max(0.0, min(100.0, effectiveness))
            
            eff_dict[layer_name] = effectiveness
    
    return eff_dict


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison_histograms(original_acts_baseline, quantized_acts_baseline, quantizers_baseline,
                               original_acts_qat, quantized_acts_qat, scales_qat, num_bits, output_dir):
    """
    Create side-by-side comparison plots for Baseline PTQ vs QAT-BR.
    
    Shows how activations align with quantization levels.
    Improved version with larger, clearer histograms.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set better matplotlib defaults for publication quality
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
    })
    
    # Get common layers
    common_layers = sorted(set(original_acts_baseline.keys()) & set(original_acts_qat.keys()))
    
    for layer_name in common_layers:
        print(f"  Plotting {layer_name}...")
        
        # Get data (move to CPU if needed)
        orig_baseline = original_acts_baseline[layer_name].cpu().numpy()
        quant_baseline = quantized_acts_baseline[layer_name].cpu().numpy()
        orig_qat = original_acts_qat[layer_name].cpu().numpy()
        quant_qat = quantized_acts_qat[layer_name].cpu().numpy()
        
        # Get quantization levels
        if layer_name in quantizers_baseline:
            levels_baseline = quantizers_baseline[layer_name].get_levels()
            if isinstance(levels_baseline, torch.Tensor):
                levels_baseline = levels_baseline.cpu().numpy()
            scale_baseline = quantizers_baseline[layer_name].scale
        else:
            continue
        
        # For QAT, construct levels from scale
        if layer_name in scales_qat:
            scale_qat = scales_qat[layer_name]
            qmax = 2 ** num_bits - 1
            levels_qat = np.arange(0, qmax + 1) * scale_qat
        else:
            continue
        
        # Determine smart x-axis limits: show 98th percentile to capture skewed Gaussian clearly
        # This shows the shape without weird zoom or cutting off too much
        xlim_max_baseline = float(np.percentile(orig_baseline, 98)) * 1.1 if len(orig_baseline) > 0 else 1.0
        xlim_max_qat = float(np.percentile(orig_qat, 98)) * 1.1 if len(orig_qat) > 0 else 1.0
        # Use the larger of the two, but cap at reasonable max
        xlim_max = min(max(xlim_max_baseline, xlim_max_qat), max(orig_baseline.max() if len(orig_baseline) > 0 else 0, 
                                                                   orig_qat.max() if len(orig_qat) > 0 else 0) * 1.05)
        
        # Create figure with 2x2 subplots - MUCH LARGER for visibility
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'{layer_name.upper()}: Baseline PTQ vs QAT-BR ({num_bits}-bit)', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        # === Baseline PTQ ===
        # Top-left: Pre-quantization (continuous, skewed Gaussian)
        ax = axes[0, 0]
        # Use fewer bins for continuous data to show shape better
        n_bins_continuous = 100
        ax.hist(orig_baseline, bins=n_bins_continuous, alpha=0.8, color='steelblue', 
                edgecolor='black', linewidth=0.5, density=False)
        # Draw quantization levels
        for level in levels_baseline:
            if level <= xlim_max:
                ax.axvline(level, color='red', linestyle='--', linewidth=2.5, alpha=0.9, zorder=10)
        ax.set_title(f'Baseline PTQ: Pre-Quantization\n(scale={scale_baseline:.4f})', 
                    fontweight='bold', fontsize=16, pad=10)
        ax.set_xlabel('Activation Value', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, xlim_max])
        # Add statistics text
        mean_b = np.mean(orig_baseline)
        std_b = np.std(orig_baseline)
        ax.text(0.98, 0.95, f'Mean: {mean_b:.3f}\nStd: {std_b:.3f}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=11)
        
        # Bottom-left: Post-quantization (discrete values - use bar-style for clarity)
        ax = axes[1, 0]
        # For discrete quantized data, count occurrences of each unique value
        unique_quant_b, counts_b = np.unique(quant_baseline, return_counts=True)
        if len(unique_quant_b) <= 20:  # For low-bit quantization, show as bar chart
            # Use bar width proportional to scale
            bar_width = scale_baseline * 0.8 if scale_baseline > 0 else 0.01
            ax.bar(unique_quant_b, counts_b, width=bar_width, 
                   alpha=0.8, color='forestgreen', edgecolor='black', linewidth=1.5)
        else:
            # Fallback to histogram if too many levels
            ax.hist(quant_baseline, bins=min(100, len(unique_quant_b)*2), alpha=0.8, 
                   color='forestgreen', edgecolor='black', linewidth=0.5, density=False)
        # Draw quantization levels
        for level in levels_baseline:
            if level <= xlim_max:
                ax.axvline(level, color='red', linestyle='--', linewidth=2.5, alpha=0.9, zorder=10)
        ax.set_title('Baseline PTQ: Post-Quantization\n(Discrete values)', 
                    fontweight='bold', fontsize=16, pad=10)
        ax.set_xlabel('Activation Value', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, xlim_max])
        
        # === QAT-BR ===
        # Top-right: Pre-quantization (BR-shaped, should cluster better)
        ax = axes[0, 1]
        ax.hist(orig_qat, bins=n_bins_continuous, alpha=0.8, color='steelblue', 
                edgecolor='black', linewidth=0.5, density=False)
        # Draw quantization levels
        for level in levels_qat:
            if level <= xlim_max:
                ax.axvline(level, color='red', linestyle='--', linewidth=2.5, alpha=0.9, zorder=10)
        ax.set_title(f'QAT-BR: Pre-Quantization\n(BR-shaped, scale={scale_qat:.4f})', 
                    fontweight='bold', fontsize=16, pad=10)
        ax.set_xlabel('Activation Value', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, xlim_max])
        # Add statistics text
        mean_q = np.mean(orig_qat)
        std_q = np.std(orig_qat)
        ax.text(0.98, 0.95, f'Mean: {mean_q:.3f}\nStd: {std_q:.3f}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=11)
        # Add BR annotation
        ax.text(0.02, 0.95, 'BR pushes toward\nred lines →',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8), fontsize=12, fontweight='bold')
        
        # Bottom-right: Post-quantization (discrete values - use bar-style)
        ax = axes[1, 1]
        # For discrete quantized data, count occurrences of each unique value
        unique_quant_q, counts_q = np.unique(quant_qat, return_counts=True)
        if len(unique_quant_q) <= 20:  # For low-bit quantization, show as bar chart
            # Use bar width proportional to scale
            bar_width = scale_qat * 0.8 if scale_qat > 0 else 0.01
            ax.bar(unique_quant_q, counts_q, width=bar_width, 
                   alpha=0.8, color='forestgreen', edgecolor='black', linewidth=1.5)
        else:
            # Fallback to histogram if too many levels
            ax.hist(quant_qat, bins=min(100, len(unique_quant_q)*2), alpha=0.8, 
                   color='forestgreen', edgecolor='black', linewidth=0.5, density=False)
        # Draw quantization levels
        for level in levels_qat:
            if level <= xlim_max:
                ax.axvline(level, color='red', linestyle='--', linewidth=2.5, alpha=0.9, zorder=10)
        ax.set_title('QAT-BR: Post-Quantization\n(Discrete values)', 
                    fontweight='bold', fontsize=16, pad=10)
        ax.set_xlabel('Activation Value', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, xlim_max])
        
        # Tight layout with padding
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save with high DPI for clarity
        save_path = os.path.join(output_dir, f'{layer_name}_comparison.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"    Saved: {layer_name}_comparison.png (improved, high-res)")


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Quantization Evaluation: Baseline PTQ vs QAT-BR')
    
    # Model paths
    parser.add_argument('--baseline-model', type=str, help='Path to baseline FP32 model checkpoint')
    parser.add_argument('--qat-model', type=str, help='Path to QAT-BR model checkpoint')
    
    # Evaluation settings
    parser.add_argument('--num-bits', type=int, default=2, help='Number of bits for quantization')
    parser.add_argument('--calibration-batches', type=int, default=50, help='Number of batches for PTQ calibration')
    parser.add_argument('--calibration-percentile', type=float, default=99.0, 
                       help='Percentile for PTQ calibration (default: 99.0, ignores outliers)')
    parser.add_argument('--ptq-calibration-method', type=str, default='percentile',
                       choices=['percentile', 'mse_search'],
                       help="PTQ calibration method. Default 'percentile' (simple/standard). 'mse_search' is stronger but unfair.")
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--mode', type=str, default='both', choices=['ptq', 'qat', 'both'],
                       help='Evaluation mode: ptq (baseline only), qat (QAT-BR only), or both')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/quantization_eval',
                       help='Directory to save results and plots')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Validation
    if args.mode in ['ptq', 'both'] and not args.baseline_model:
        parser.error("--baseline-model required for PTQ evaluation")
    if args.mode in ['qat', 'both'] and not args.qat_model:
        parser.error("--qat-model required for QAT evaluation")
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print("=" * 80)
    print("QUANTIZATION EVALUATION: Baseline PTQ vs QAT-BR")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Quantization: {args.num_bits}-bit")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Data transforms - different for baseline vs QAT!
    # Baseline uses float32 [0, 1]
    transform_baseline = transforms.Compose([transforms.ToTensor()])
    # QAT uses uint8 [0, 255]
    transform_qat = transforms.Lambda(lambda x: (transforms.ToTensor()(x) * 255).to(torch.uint8))
    
    # We'll load both datasets
    test_dataset_baseline = datasets.MNIST(root='./data', train=False, download=True, transform=transform_baseline)
    test_dataset_qat = datasets.MNIST(root='./data', train=False, download=True, transform=transform_qat)
    
    results = {}
    
    # ========== BASELINE PTQ ==========
    if args.mode in ['ptq', 'both']:
        print("\n" + "=" * 80)
        print("BASELINE: Post-Training Quantization (PTQ)")
        print("=" * 80)
        
        # Load baseline model
        print(f"\nLoading baseline model from: {args.baseline_model}")
        checkpoint = torch.load(args.baseline_model, map_location=device)
        # Read clip_value from checkpoint (None = standard ReLU, 1.0 = ReLU1, etc.)
        clip_value_baseline = checkpoint.get('clip_value', None)
        baseline_model = PlainConvFlatten(input_channels=1, num_classes=10, base=16, clip_value=clip_value_baseline).to(device)
        if 'model_state_dict' in checkpoint:
            baseline_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            baseline_model.load_state_dict(checkpoint)
        baseline_model.eval()
        print(f"✓ Baseline model loaded (clip_value={clip_value_baseline if clip_value_baseline is not None else 'None (standard ReLU)'})")
        
        # Create data loaders for baseline (float32 [0, 1])
        baseline_test_loader = DataLoader(test_dataset_baseline, batch_size=args.batch_size, shuffle=False)
        baseline_calib_loader = DataLoader(test_dataset_baseline, batch_size=args.batch_size, shuffle=True)
        
        # First, test FP32 accuracy (no quantization)
        print(f"\nStep 1: Testing baseline FP32 (full precision) accuracy...")
        correct_fp32 = 0
        total_fp32 = 0
        with torch.no_grad():
            for data, target in baseline_test_loader:
                data, target = data.to(device), target.to(device)
                output = baseline_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct_fp32 += pred.eq(target.view_as(pred)).sum().item()
                total_fp32 += len(target)
        fp32_accuracy = 100. * correct_fp32 / total_fp32
        print(f"✓ Baseline FP32 Test Accuracy: {fp32_accuracy:.2f}%")
        
        # Collect activations for calibration
        print(f"\nStep 2: Collecting activations for calibration ({args.calibration_batches} batches)...")
        calib_acts = collect_activations_for_calibration(baseline_model, baseline_calib_loader, device, args.calibration_batches)
        
        # Calibrate quantizers
        print(f"\nStep 3: Calibrating PTQ quantizers for {args.num_bits}-bit...")
        quantizers_baseline = calibrate_ptq_quantizers(
            calib_acts,
            args.num_bits,
            percentile=args.calibration_percentile,
            method=args.ptq_calibration_method,
        )
        
        # Apply PTQ and evaluate
        print(f"\nStep 4: Applying INT{args.num_bits} PTQ quantization and evaluating...")
        print(f"       (Activations will be quantized during inference)")
        baseline_acc, baseline_quant_acts, baseline_orig_acts = apply_ptq_quantization(
            baseline_model, quantizers_baseline, baseline_test_loader, device
        )
        
        # Compute metrics
        baseline_mse = compute_quantization_mse(baseline_orig_acts, baseline_quant_acts)
        baseline_eff = compute_clustering_effectiveness(baseline_orig_acts, baseline_quant_acts)
        
        results['baseline_ptq'] = {
            'fp32_accuracy': fp32_accuracy,
            'accuracy': baseline_acc,
            'mse': baseline_mse,
            'effectiveness': baseline_eff,
            'quantized_acts': baseline_quant_acts,
            'original_acts': baseline_orig_acts,
            'quantizers': quantizers_baseline
        }
        
        print(f"\n{'='*80}")
        print("BASELINE PTQ RESULTS:")
        print(f"{'='*80}")
        print(f"FP32 Test Accuracy:  {fp32_accuracy:.2f}%")
        print(f"INT{args.num_bits} Test Accuracy:  {baseline_acc:.2f}%")
        print(f"Accuracy Drop:       {fp32_accuracy - baseline_acc:.2f}% (due to quantization)")
        print(f"\nPer-Layer Quantization MSE:")
        for layer, mse in baseline_mse.items():
            eff = baseline_eff.get(layer, 0)
            print(f"  {layer}: MSE={mse:.6f}, Effectiveness={eff:.1f}%")
        print(f"\nAverage MSE: {np.mean(list(baseline_mse.values())):.6f}")
        print(f"Average Effectiveness: {np.mean(list(baseline_eff.values())):.1f}%")
    
    # ========== QAT-BR ==========
    if args.mode in ['qat', 'both']:
        print("\n" + "=" * 80)
        print("QAT-BR: Quantization-Aware Training with Bin Regularization")
        print("=" * 80)
        
        # Load QAT model
        print(f"\nLoading QAT-BR model from: {args.qat_model}")
        qat_model = PlainConvFlattenQAT(input_channels=1, num_classes=10, base=16, clip_value=1.0, num_bits=args.num_bits).to(device)
        checkpoint = torch.load(args.qat_model, map_location=device)
        if 'model_state_dict' in checkpoint:
            qat_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            qat_model.load_state_dict(checkpoint)
        qat_model.eval()
        print("✓ QAT-BR model loaded")
        
        # Create data loader for QAT (uint8 [0, 255])
        qat_test_loader = DataLoader(test_dataset_qat, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate (uses learned scales from training)
        print(f"\nEvaluating QAT-BR model with learned scales...")
        qat_acc, qat_quant_acts, qat_orig_acts, qat_scales = evaluate_qat_model(
            qat_model, qat_test_loader, device
        )
        
        # Compute metrics
        qat_mse = compute_quantization_mse(qat_orig_acts, qat_quant_acts)
        qat_eff = compute_clustering_effectiveness(qat_orig_acts, qat_quant_acts)
        
        results['qat_br'] = {
            'accuracy': qat_acc,
            'mse': qat_mse,
            'effectiveness': qat_eff,
            'quantized_acts': qat_quant_acts,
            'original_acts': qat_orig_acts,
            'scales': qat_scales
        }
        
        print(f"\n{'='*80}")
        print("QAT-BR RESULTS:")
        print(f"{'='*80}")
        print(f"Test Accuracy: {qat_acc:.2f}%")
        print(f"\nPer-Layer Quantization MSE:")
        for layer, mse in qat_mse.items():
            eff = qat_eff.get(layer, 0)
            print(f"  {layer}: MSE={mse:.6f}, Effectiveness={eff:.1f}%")
        print(f"\nAverage MSE: {np.mean(list(qat_mse.values())):.6f}")
        print(f"Average Effectiveness: {np.mean(list(qat_eff.values())):.1f}%")
    
    # ========== COMPARISON ==========
    if args.mode == 'both':
        print("\n" + "=" * 80)
        print("COMPARISON: Baseline PTQ vs QAT-BR")
        print("=" * 80)
        
        baseline_avg_mse = np.mean(list(results['baseline_ptq']['mse'].values()))
        qat_avg_mse = np.mean(list(results['qat_br']['mse'].values()))
        mse_improvement = (baseline_avg_mse - qat_avg_mse) / baseline_avg_mse * 100
        
        baseline_avg_eff = np.mean(list(results['baseline_ptq']['effectiveness'].values()))
        qat_avg_eff = np.mean(list(results['qat_br']['effectiveness'].values()))
        
        fp32_accuracy = results['baseline_ptq']['fp32_accuracy']
        baseline_acc = results['baseline_ptq']['accuracy']
        qat_acc = results['qat_br']['accuracy']
        acc_diff = qat_acc - baseline_acc
        
        print(f"\nINT{args.num_bits} Quantized Accuracy:")
        print(f"  Baseline FP32:     {fp32_accuracy:.2f}% (before quantization)")
        print(f"  Baseline PTQ:      {baseline_acc:.2f}% (PTQ quantized)")
        print(f"  QAT-BR:            {qat_acc:.2f}% (QAT quantized)")
        print(f"  ")
        print(f"  PTQ Accuracy Drop: {fp32_accuracy - baseline_acc:.2f}%")
        print(f"  QAT-BR vs PTQ:     {acc_diff:+.2f}%")
        
        print(f"\nQuantization MSE:")
        print(f"  Baseline PTQ: {baseline_avg_mse:.6f}")
        print(f"  QAT-BR:       {qat_avg_mse:.6f}")
        print(f"  Improvement:  {mse_improvement:.1f}% reduction")
        
        print(f"\nClustering Effectiveness:")
        print(f"  Baseline PTQ: {baseline_avg_eff:.1f}%")
        print(f"  QAT-BR:       {qat_avg_eff:.1f}%")
        print(f"  Improvement:  {qat_avg_eff - baseline_avg_eff:+.1f}%")
        
        # Generate comparison plots
        print(f"\nGenerating comparison plots...")
        plot_comparison_histograms(
            results['baseline_ptq']['original_acts'],
            results['baseline_ptq']['quantized_acts'],
            results['baseline_ptq']['quantizers'],
            results['qat_br']['original_acts'],
            results['qat_br']['quantized_acts'],
            results['qat_br']['scales'],
            args.num_bits,
            args.output_dir
        )
        
        # Save summary
        summary_path = os.path.join(args.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("QUANTIZATION EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Quantization: {args.num_bits}-bit\n\n")
            f.write(f"Accuracy:\n")
            f.write(f"  Baseline FP32:     {fp32_accuracy:.2f}% (before quantization)\n")
            f.write(f"  Baseline PTQ:      {baseline_acc:.2f}% (PTQ quantized)\n")
            f.write(f"  QAT-BR:            {qat_acc:.2f}% (QAT quantized)\n\n")
            f.write(f"  PTQ Accuracy Drop: {fp32_accuracy - baseline_acc:.2f}%\n")
            f.write(f"  QAT-BR vs PTQ:     {acc_diff:+.2f}%\n\n")
            f.write(f"Quantization MSE:\n")
            f.write(f"  Baseline PTQ: {baseline_avg_mse:.6f}\n")
            f.write(f"  QAT-BR:       {qat_avg_mse:.6f}\n")
            f.write(f"  Improvement:  {mse_improvement:.1f}% reduction\n\n")
            f.write(f"Clustering Effectiveness:\n")
            f.write(f"  Baseline PTQ: {baseline_avg_eff:.1f}%\n")
            f.write(f"  QAT-BR:       {qat_avg_eff:.1f}%\n")
            f.write(f"  Improvement:  {qat_avg_eff - baseline_avg_eff:+.1f}%\n")
        print(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

