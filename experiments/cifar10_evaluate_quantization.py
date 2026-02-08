#!/usr/bin/env python3
"""
Comprehensive Quantization Evaluation: Baseline PTQ vs QAT-BR (CIFAR-10)

This script performs a fair comparison between:
1. Baseline (FP32 trained) + PTQ with calibration
2. QAT-BR (trained with BR) using learned scales

The goal is to show that BR reduces quantization error and maintains accuracy.

Usage:
    # Evaluate both baseline PTQ and QAT-BR
    python experiments/cifar10_evaluate_quantization.py \
        --baseline-model checkpoints/cifar10_simple_baseline_*.pth \
        --qat-model checkpoints/cifar10_qat_binreg_*.pth \
        --num-bits 2 \
        --output-dir results/cifar10_quantization_comparison

    # Baseline PTQ only
    python experiments/cifar10_evaluate_quantization.py \
        --baseline-model checkpoints/cifar10_simple_baseline_*.pth \
        --num-bits 2 \
        --mode ptq

    # QAT-BR evaluation only
    python experiments/cifar10_evaluate_quantization.py \
        --qat-model checkpoints/cifar10_qat_binreg_*.pth \
        --num-bits 2 \
        --mode qat
"""

import os
import sys
import argparse
import random
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
from experiments.cifar10_baseline import SimpleCNN_CIFAR10
from experiments.cifar10_qat_binreg import SimpleCNN_CIFAR10_QAT

# Import ResNet18 for CIFAR-10
try:
    from experiments.cifar10_resnet18_baseline import get_resnet18_cifar10
    RESNET18_AVAILABLE = True
except ImportError:
    RESNET18_AVAILABLE = False
    print("Warning: ResNet18 not available. Only SimpleCNN supported.")


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
        Memory-efficient: works with large tensors by using numpy.
        """
        # Ensure on CPU and flatten
        if activations.is_cuda:
            acts_flat = activations.cpu().flatten()
        else:
            acts_flat = activations.flatten()
        
        # Convert to numpy for percentile (more memory efficient for large arrays)
        # Use float32 to save memory
        acts_np = acts_flat.numpy().astype(np.float32)
        
        # Compute percentile
        max_val = float(np.percentile(np.abs(acts_np), percentile))
        
        if max_val == 0:
            self.scale = 1.0
        else:
            self.scale = max_val / self.qmax
        self.is_calibrated = True
        actual_max = float(np.max(np.abs(acts_np)))
        print(f"  Calibrated: {percentile}th percentile={max_val:.4f}, actual_max={actual_max:.4f}")
        print(f"             scale={self.scale:.6f}, levels=[0, {self.scale:.4f}, ..., {self.qmax*self.scale:.4f}]")
        
        # Clear memory
        del acts_np, acts_flat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def calibrate_mse_search(
        self,
        activations: torch.Tensor,
        candidate_percentiles=(50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5, 99.0, 99.5, 99.9),
        sample_size: int = 200_000,
        rng: "torch.Generator | None" = None,
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
            # Use a CPU generator for reproducibility across devices
            idx = torch.randint(0, acts.numel(), (sample_size,), generator=rng, device='cpu')
            acts = acts[idx.to(acts.device)]

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


def collect_activations_for_calibration(model, loader, device, num_batches=50, max_samples_per_layer=500000, rng: "torch.Generator | None" = None):
    """
    Collect activations from ReLU layers for PTQ calibration.
    
    Memory-efficient version: samples activations instead of storing all.
    
    Args:
        max_samples_per_layer: Maximum samples to keep per layer (prevents OOM)
    
    Returns dict of {layer_name: sampled_activations}
    """
    model.eval()
    activations = {}
    sample_counts = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
                sample_counts[name] = 0
            
            # Memory-efficient: only keep samples if under limit
            if sample_counts[name] < max_samples_per_layer:
                output_flat = output.detach().cpu().flatten()
                needed = max_samples_per_layer - sample_counts[name]
                
                if len(output_flat) <= needed:
                    # Can fit all
                    activations[name].append(output_flat)
                    sample_counts[name] += len(output_flat)
                else:
                    # Sample randomly
                    indices = torch.randperm(len(output_flat), generator=rng)[:needed]
                    activations[name].append(output_flat[indices])
                    sample_counts[name] += needed
        return hook
    
    # Import ClippedReLU from both baseline scripts
    from experiments.cifar10_baseline import ClippedReLU as ClippedReLU_baseline
    try:
        from experiments.cifar10_resnet18_baseline import ClippedReLU as ClippedReLU_resnet
    except ImportError:
        ClippedReLU_resnet = ClippedReLU_baseline  # Fallback
    
    # Try to import ClippedReLU from MNIST scripts
    ClippedReLU_mnist = None
    try:
        from experiments.mnist_baseline import ClippedReLU as ClippedReLU_mnist
    except ImportError:
        try:
            # Also check if it's defined in the current module (for mnist_automated_ptq_sweep)
            import sys
            for module_name, module_obj in sys.modules.items():
                if hasattr(module_obj, 'ClippedReLU'):
                    ClippedReLU_mnist = getattr(module_obj, 'ClippedReLU')
                    break
        except:
            pass
    
    # Build list of recognized ReLU types
    relu_types = [nn.ReLU, nn.ReLU6, ClippedReLU_baseline, ClippedReLU_resnet]
    if ClippedReLU_mnist is not None:
        relu_types.append(ClippedReLU_mnist)
    
    # Also check for any module with clip_value attribute (generic ClippedReLU detection)
    def is_clipped_relu(module):
        return (hasattr(module, 'clip_value') and 
                hasattr(module, 'forward') and
                isinstance(module, nn.Module))
    
    # Register hooks on all ReLU/clamp operations
    hooks = []
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, tuple(relu_types)) or is_clipped_relu(module):
            layer_name = f"relu{layer_count + 1}"
            hooks.append(module.register_forward_hook(make_hook(layer_name)))
            layer_count += 1
    
    print(f"Collecting activations from {layer_count} ReLU layers for calibration...")
    print(f"  Memory limit: {max_samples_per_layer:,} samples per layer")
    
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
            
            # Clear GPU cache periodically
            if (batch_idx + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate activations across batches
    for name in activations:
        if activations[name]:
            activations[name] = torch.cat(activations[name])
            print(f"  {name}: {len(activations[name]):,} samples")
        else:
            activations[name] = torch.tensor([])
    
    print(f"Collected activations for {len(activations)} layers")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return activations


def calibrate_ptq_quantizers(activations_dict, num_bits, percentile=99.0, method: str = "percentile", rng: "torch.Generator | None" = None):
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
            q.calibrate_mse_search(acts, rng=rng)
        else:
            raise ValueError(f"Unknown PTQ calibration method: {method}")
        quantizers[layer_name] = q
    return quantizers


def apply_ptq_quantization(model, quantizers, loader, device, rng: "torch.Generator | None" = None):
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
    
    # Import ClippedReLU from both baseline scripts
    from experiments.cifar10_baseline import ClippedReLU as ClippedReLU_baseline
    try:
        from experiments.cifar10_resnet18_baseline import ClippedReLU as ClippedReLU_resnet
    except ImportError:
        ClippedReLU_resnet = ClippedReLU_baseline  # Fallback
    
    # Try to import ClippedReLU from MNIST scripts
    ClippedReLU_mnist = None
    try:
        from experiments.mnist_baseline import ClippedReLU as ClippedReLU_mnist
    except ImportError:
        try:
            # Also check if it's defined in the current module (for mnist_automated_ptq_sweep)
            import sys
            for module_name, module_obj in sys.modules.items():
                if hasattr(module_obj, 'ClippedReLU'):
                    ClippedReLU_mnist = getattr(module_obj, 'ClippedReLU')
                    break
        except:
            pass
    
    # Build list of recognized ReLU types
    relu_types = [nn.ReLU, nn.ReLU6, ClippedReLU_baseline, ClippedReLU_resnet]
    if ClippedReLU_mnist is not None:
        relu_types.append(ClippedReLU_mnist)
    
    # Also check for any module with clip_value attribute (generic ClippedReLU detection)
    def is_clipped_relu(module):
        return (hasattr(module, 'clip_value') and 
                hasattr(module, 'forward') and
                isinstance(module, nn.Module))
    
    # Wrap each ReLU with a quantizer
    relu_modules = []
    original_forwards = []
    layer_idx = 0
    batch_counter = [0]  # Mutable to share across closures
    sanity = {
        'layers': {},
        'num_batches': 0,
    }
    
    for name, module in model.named_modules():
        if isinstance(module, tuple(relu_types)) or is_clipped_relu(module):
            layer_name = f"relu{layer_idx + 1}"
            if layer_name in quantizers:
                relu_modules.append((module, layer_name, quantizers[layer_name]))
                original_forwards.append(module.forward)
                
                # Create new forward that quantizes
                def make_quantized_forward(original_forward, layer_name, quantizer, acts_orig, acts_quant, batch_counter, rng):
                    def quantized_forward(x):
                        # Apply original activation
                        output = original_forward(x)
                        # Store original (only first 3 batches, flatten immediately to save memory)
                        if batch_counter[0] < 3:
                            acts_orig[layer_name].append(output.detach().cpu().flatten())
                        # Quantize (and keep integer codes for sanity checks)
                        x_int = quantizer.quantize_int(output)
                        output_quant = x_int * quantizer.scale
                        # Store quantized (only first 3 batches, flatten immediately)
                        if batch_counter[0] < 3:
                            acts_quant[layer_name].append(output_quant.detach().cpu().flatten())
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
                                idx = torch.randint(0, flat.numel(), (200_000,), generator=rng, device='cpu')
                                flat = flat[idx.to(flat.device)]
                            layer_s['example_unique_codes'] = torch.unique(flat).detach().cpu().tolist()
                        # Return quantized (this actually flows through network!)
                        return output_quant
                    return quantized_forward
                
                # Replace forward method
                module.forward = make_quantized_forward(
                    original_forwards[-1], layer_name, quantizers[layer_name],
                    original_acts_all, quantized_acts_all, batch_counter, rng
                )
            layer_idx += 1
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            # Update batch counter for activation collection
            batch_counter[0] = batch_idx
            
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
    
    # Concatenate activations (collected from first 3 batches, already flattened)
    print(f"\nConcatenating collected activations...")
    quantized_acts = {}
    original_acts = {}
    for name, acts in quantized_acts_all.items():
        if acts:  # Check if list is not empty
            quantized_acts[name] = torch.cat(acts)  # Already flattened
            print(f"  {name}: {len(quantized_acts[name]):,} samples")
        else:
            print(f"  {name}: NO SAMPLES (error!)")
            quantized_acts[name] = torch.tensor([])  # Empty tensor if no acts collected
    
    for name, acts in original_acts_all.items():
        if acts:
            original_acts[name] = torch.cat(acts)  # Already flattened
        else:
            original_acts[name] = torch.tensor([])
    
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
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # Create figure with 2x2 subplots - RAW COUNTS (like TensorBoard)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{layer_name}: Baseline PTQ vs QAT-BR ({num_bits}-bit)', fontsize=16, fontweight='bold')
        
        # Use raw data (no filtering)
        orig_b, quant_b = orig_baseline, quant_baseline
        orig_q, quant_q = orig_qat, quant_qat
        
        # Helper function to plot histograms with raw counts (like TensorBoard)
        def plot_histograms(axes_set):
            use_log = False  # Standard linear scale
            
            # === Baseline PTQ ===
            # Top-left: Pre-quantization
            ax = axes_set[0, 0]
            ax.hist(orig_b, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
            for level in levels_baseline:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title(f'Baseline PTQ: Pre-Quantization\n(scale={scale_baseline:.4f})', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            # Set x-axis to actual data range (like TensorBoard)
            if len(orig_b) > 0:
                ax.set_xlim([0, orig_b.max() * 1.05])
            
            # Bottom-left: Post-quantization
            ax = axes_set[1, 0]
            ax.hist(quant_b, bins=200, alpha=0.7, color='green', edgecolor='black', density=False)
            for level in levels_baseline:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title('Baseline PTQ: Post-Quantization\n(Discrete values)', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            # Set x-axis to actual data range (like TensorBoard)
            if len(quant_b) > 0:
                ax.set_xlim([0, quant_b.max() * 1.05])
            
            # === QAT-BR ===
            # Top-right: Pre-quantization
            ax = axes_set[0, 1]
            ax.hist(orig_q, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
            for level in levels_qat:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title(f'QAT-BR: Pre-Quantization\n(BR-shaped, scale={scale_qat:.4f})', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            # Set x-axis to actual data range (like TensorBoard)
            if len(orig_q) > 0:
                ax.set_xlim([0, orig_q.max() * 1.05])
            ax.text(0.02, 0.95, 'BR pushes toward red lines →',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=9)
            
            # Bottom-right: Post-quantization
            ax = axes_set[1, 1]
            ax.hist(quant_q, bins=200, alpha=0.7, color='green', edgecolor='black', density=False)
            for level in levels_qat:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title('QAT-BR: Post-Quantization\n(Discrete values)', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            # Set x-axis to actual data range (like TensorBoard)
            if len(quant_q) > 0:
                ax.set_xlim([0, quant_q.max() * 1.05])
        
        # Plot histograms (raw counts, like TensorBoard)
        plot_histograms(axes)
        
        # Tight layout
        fig.tight_layout()
        
        # Save
        save_path = os.path.join(output_dir, f'{layer_name}_comparison.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # ===== ZOOMED VERSION: Focus on where DATA actually is =====
        fig_zoom, axes_zoom = plt.subplots(2, 2, figsize=(16, 12))
        fig_zoom.suptitle(f'{layer_name.upper()} - ZOOMED (Where the Data Is)', fontsize=16, fontweight='bold')
        
        # Zoom to where most data is (95th percentile of baseline, which has wider distribution)
        # This shows QAT-BR's tight clustering vs baseline's spread
        zoom_max = float(np.percentile(orig_b, 95)) * 1.2  # Add 20% margin
        
        def plot_histograms_zoomed(axes_set):
            # === Baseline PTQ ===
            # Top-left: Pre-quantization
            ax = axes_set[0, 0]
            ax.hist(orig_b, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
            for level in levels_baseline:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title(f'Baseline PTQ: Pre-Quantization\n(scale={scale_baseline:.4f})', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, zoom_max])
            
            # Bottom-left: Post-quantization
            ax = axes_set[1, 0]
            ax.hist(quant_b, bins=200, alpha=0.7, color='green', edgecolor='black', density=False)
            for level in levels_baseline:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title('Baseline PTQ: Post-Quantization\n(Discrete values)', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, zoom_max])
            
            # === QAT-BR ===
            # Top-right: Pre-quantization
            ax = axes_set[0, 1]
            ax.hist(orig_q, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
            for level in levels_qat:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title(f'QAT-BR: Pre-Quantization\n(BR-shaped, scale={scale_qat:.4f})', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, zoom_max])
            ax.text(0.02, 0.95, 'BR pushes toward red lines →',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=9)
            
            # Bottom-right: Post-quantization
            ax = axes_set[1, 1]
            ax.hist(quant_q, bins=200, alpha=0.7, color='green', edgecolor='black', density=False)
            for level in levels_qat:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title('QAT-BR: Post-Quantization\n(Discrete values)', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, zoom_max])
        
        plot_histograms_zoomed(axes_zoom)
        plt.tight_layout()
        
        save_path_zoom = os.path.join(output_dir, f'{layer_name}_comparison_ZOOMED.png')
        fig_zoom.savefig(save_path_zoom, dpi=150, bbox_inches='tight')
        plt.close(fig_zoom)
        
        # ===== LOG SCALE VERSION: Better visualization of long tails =====
        fig_log, axes_log = plt.subplots(2, 2, figsize=(16, 12))
        fig_log.suptitle(f'{layer_name.upper()} - LOG SCALE (Full Range)', fontsize=16, fontweight='bold')
        
        def plot_histograms_log(axes_set):
            # === Baseline PTQ ===
            # Top-left: Pre-quantization
            ax = axes_set[0, 0]
            ax.hist(orig_b, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
            for level in levels_baseline:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title(f'Baseline PTQ: Pre-Quantization\n(scale={scale_baseline:.4f})', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count (log scale)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            if len(orig_b) > 0:
                ax.set_xlim([0, orig_b.max() * 1.05])
            
            # Bottom-left: Post-quantization
            ax = axes_set[1, 0]
            ax.hist(quant_b, bins=200, alpha=0.7, color='green', edgecolor='black', density=False)
            for level in levels_baseline:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title('Baseline PTQ: Post-Quantization\n(Discrete values)', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count (log scale)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            if len(quant_b) > 0:
                ax.set_xlim([0, quant_b.max() * 1.05])
            
            # === QAT-BR ===
            # Top-right: Pre-quantization
            ax = axes_set[0, 1]
            ax.hist(orig_q, bins=200, alpha=0.7, color='blue', edgecolor='black', density=False)
            for level in levels_qat:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title(f'QAT-BR: Pre-Quantization\n(BR-shaped, scale={scale_qat:.4f})', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count (log scale)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            if len(orig_q) > 0:
                ax.set_xlim([0, orig_q.max() * 1.05])
            ax.text(0.02, 0.95, 'BR pushes toward red lines →',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=9)
            
            # Bottom-right: Post-quantization
            ax = axes_set[1, 1]
            ax.hist(quant_q, bins=200, alpha=0.7, color='green', edgecolor='black', density=False)
            for level in levels_qat:
                ax.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_title('QAT-BR: Post-Quantization\n(Discrete values)', fontweight='bold', fontsize=11)
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Count (log scale)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            if len(quant_q) > 0:
                ax.set_xlim([0, quant_q.max() * 1.05])
        
        plot_histograms_log(axes_log)
        plt.tight_layout()
        
        save_path_log = os.path.join(output_dir, f'{layer_name}_comparison_LOG.png')
        fig_log.savefig(save_path_log, dpi=150, bbox_inches='tight')
        plt.close(fig_log)
        
        print(f"    Saved 3 versions: {layer_name}_comparison.png (full), _ZOOMED.png (3x quant range), _LOG.png (log scale)")


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Quantization Evaluation: Baseline PTQ vs QAT-BR')
    
    # Model paths
    parser.add_argument('--baseline-model', type=str, help='Path to baseline FP32 model checkpoint')
    parser.add_argument('--qat-model', type=str, help='Path to QAT-BR model checkpoint')
    
    # Model type
    parser.add_argument('--model-type', type=str, default='auto', choices=['auto', 'simplecnn', 'resnet18'],
                       help='Model architecture type. "auto" tries to detect from checkpoint, "simplecnn" or "resnet18" to force.')
    
    # Evaluation settings
    parser.add_argument('--num-bits', type=int, default=2, help='Number of bits for quantization')
    parser.add_argument('--calibration-batches', type=int, default=50, help='Number of batches for PTQ calibration')
    parser.add_argument('--calibration-percentile', type=float, default=99.0, 
                       help='Percentile for PTQ calibration (default: 99.0, ignores outliers)')
    parser.add_argument('--ptq-calibration-method', type=str, default='percentile',
                       choices=['percentile', 'mse_search'],
                       help="PTQ calibration method. Default 'percentile' (simple/standard). 'mse_search' is stronger but unfair.")
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (default: 0 for reproducibility)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True,
                       help='Enable deterministic kernels (default: True). Use --no-deterministic to disable.')
    parser.add_argument('--mode', type=str, default='both', choices=['ptq', 'qat', 'both'],
                       help='Evaluation mode: ptq (baseline only), qat (QAT-BR only), or both')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/cifar10_quantization_eval',
                       help='Directory to save results and plots')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()

    # Reproducibility (sampling + deterministic kernels)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    # Dedicated RNGs so DataLoader shuffling doesn't affect activation sampling
    loader_rng = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    sample_rng = torch.Generator(device='cpu').manual_seed(args.seed + 2)
    
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
    print(f"Seed: {args.seed} | Deterministic: {args.deterministic} | num_workers: {args.num_workers}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Data transforms - different for baseline vs QAT!
    # Baseline uses float32 [0, 1] with normalization
    transform_baseline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # QAT uses same normalization (already applied in training)
    transform_qat = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # We'll load both datasets
    test_dataset_baseline = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_baseline)
    test_dataset_qat = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_qat)
    
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
        
        # Determine model type
        model_type = args.model_type
        if model_type == 'auto':
            # Try to auto-detect from checkpoint or state dict keys
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # ResNet18 has 'conv1.weight', 'layer1.0.conv1.weight', etc.
            # SimpleCNN has 'conv1.weight', 'conv2.weight', etc. (no 'layer1')
            if any('layer1' in key for key in state_dict.keys()):
                model_type = 'resnet18'
                print("  Auto-detected: ResNet18 (found 'layer1' in state dict)")
            else:
                model_type = 'simplecnn'
                print("  Auto-detected: SimpleCNN")
        
        # Create appropriate model
        if model_type == 'resnet18':
            if not RESNET18_AVAILABLE:
                raise ImportError("ResNet18 not available. Install or check imports.")
            pretrained = checkpoint.get('pretrained', False)
            baseline_model = get_resnet18_cifar10(
                num_classes=10, 
                pretrained=pretrained,
                clip_value=clip_value_baseline
            ).to(device)
            print(f"  Created ResNet18 (pretrained={pretrained})")
        else:  # simplecnn
            baseline_model = SimpleCNN_CIFAR10(num_classes=10, base=32, clip_value=clip_value_baseline).to(device)
            print(f"  Created SimpleCNN")
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            baseline_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            baseline_model.load_state_dict(checkpoint)
        baseline_model.eval()
        print(f"✓ Baseline model loaded (type={model_type}, clip_value={clip_value_baseline if clip_value_baseline is not None else 'None (standard ReLU)'})")
        
        # Create data loaders for baseline (float32 [0, 1])
        baseline_test_loader = DataLoader(
            test_dataset_baseline, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        baseline_calib_loader = DataLoader(
            test_dataset_baseline,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=loader_rng,
        )
        
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
        # For ResNet18, use fewer batches and sample activations to prevent OOM
        if model_type == 'resnet18':
            # ResNet18 is larger, so use fewer batches and sample activations
            effective_batches = min(args.calibration_batches, 20)  # Cap at 20 for ResNet18
            max_samples = 500000  # 500k samples per layer
            print(f"\nStep 2: Collecting activations for calibration ({effective_batches} batches, max {max_samples:,} samples/layer)...")
        else:
            effective_batches = args.calibration_batches
            max_samples = 1000000  # 1M samples for smaller models
            print(f"\nStep 2: Collecting activations for calibration ({effective_batches} batches, max {max_samples:,} samples/layer)...")
        
        calib_acts = collect_activations_for_calibration(
            baseline_model, baseline_calib_loader, device, 
            num_batches=effective_batches,
            max_samples_per_layer=max_samples,
            rng=sample_rng,
        )
        
        # Calibrate quantizers
        print(f"\nStep 3: Calibrating PTQ quantizers for {args.num_bits}-bit...")
        quantizers_baseline = calibrate_ptq_quantizers(
            calib_acts,
            args.num_bits,
            percentile=args.calibration_percentile,
            method=args.ptq_calibration_method,
            rng=sample_rng,
        )
        
        # Apply PTQ and evaluate
        print(f"\nStep 4: Applying INT{args.num_bits} PTQ quantization and evaluating...")
        print(f"       (Activations will be quantized during inference)")
        baseline_acc, baseline_quant_acts, baseline_orig_acts = apply_ptq_quantization(
            baseline_model, quantizers_baseline, baseline_test_loader, device, rng=sample_rng
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
        qat_model = SimpleCNN_CIFAR10_QAT(num_classes=10, base=32, clip_value=1.0, num_bits=args.num_bits).to(device)
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




