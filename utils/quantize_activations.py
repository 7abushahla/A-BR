"""
Symmetric Uniform Quantization of Activations (PTQ)

This script:
1. Loads a trained model (baseline or A-KURE)
2. Calibrates quantization scales on a representative dataset
3. Quantizes activations at different bit-widths (2, 4, 6, 8)
4. Computes MSE per layer to evaluate quantization quality

Usage:
    python quantize_activations.py --model checkpoints/baseline.pth --bits 2 4 6 8
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from experiments.mnist_wasserstein import PlainConvFlattenReLU6
from experiments.mnist_baseline import PlainConvFlatten


class UnsignedUniformQuantizer:
    """
    Unsigned uniform quantizer for ReLU activations.
    
    Range: [0, max_val]
    Maps to integer levels [0, 2^n - 1] (matches LSQ training).
    """
    
    def __init__(self, num_bits: int):
        self.num_bits = num_bits
        self.scale = None
        self.max_val = None
        
        # Quantization levels (unsigned)
        self.qmin = 0
        self.qmax = 2 ** num_bits - 1
    
    def calibrate(self, activations: torch.Tensor):
        """Calibrate scale based on max activation value."""
        self.max_val = torch.max(activations).item()
        self.scale = self.max_val / self.qmax
        if self.scale == 0:
            self.scale = 1.0
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activations to integer levels."""
        if self.scale is None:
            raise ValueError("Must calibrate before quantizing!")
        q = torch.round(x / self.scale)
        q = torch.clamp(q, self.qmin, self.qmax)
        return q
    
    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        """Dequantize back to float."""
        return q * self.scale
    
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Full quantization cycle."""
        return self.dequantize(self.quantize(x))


class SymmetricUniformQuantizer:
    """
    Symmetric uniform quantizer for activations.
    
    Range: [-max_abs, max_abs]
    No zero-point (z=0), only scale parameter.
    """
    
    def __init__(self, num_bits: int):
        self.num_bits = num_bits
        self.scale = None
        self.max_abs = None
        
        # Quantization levels
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = 2 ** (num_bits - 1) - 1
    
    def calibrate(self, activations: torch.Tensor):
        """
        Calibrate scale based on activation statistics.
        
        Args:
            activations: Tensor of activations from calibration data
        """
        # Find max absolute value
        self.max_abs = torch.max(torch.abs(activations)).item()
        
        # Compute scale
        if self.num_bits == 1:
            self.scale = self.max_abs
        else:
            self.scale = self.max_abs / (2 ** (self.num_bits - 1) - 1)
        
        # Avoid division by zero
        if self.scale == 0:
            self.scale = 1.0
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize activations.
        
        Args:
            x: Input activations
            
        Returns:
            Quantized values (still in integer representation)
        """
        if self.scale is None:
            raise ValueError("Must calibrate before quantizing!")
        
        # Scale and round
        q = torch.round(x / self.scale)
        
        # Clip to valid range
        q = torch.clamp(q, self.qmin, self.qmax)
        
        return q
    
    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        """
        Dequantize back to float.
        
        Args:
            q: Quantized values
            
        Returns:
            Dequantized float values
        """
        return q * self.scale
    
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full quantization-dequantization cycle.
        
        Args:
            x: Input activations
            
        Returns:
            Dequantized activations (float, but quantized)
        """
        q = self.quantize(x)
        return self.dequantize(q)
    
    def __repr__(self):
        return f"SymmetricUniformQuantizer(bits={self.num_bits}, scale={self.scale:.6f}, max_abs={self.max_abs:.4f})"


def collect_activations(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    layer_names: List[str]
) -> Dict[str, List[torch.Tensor]]:
    """
    Collect activations from specified layers.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for calibration
        device: Device to run on
        layer_names: List of layer names to hook
        
    Returns:
        Dictionary mapping layer names to list of activation tensors
    """
    model.eval()
    activations = {name: [] for name in layer_names}
    
    # Register hooks
    handles = []
    
    def make_hook(name):
        def hook(module, input, output):
            activations[name].append(output.detach().cpu())
        return hook
    
    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
    
    # Collect activations
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            _ = model(data)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return activations


def calibrate_quantizers(
    activations: Dict[str, List[torch.Tensor]],
    num_bits: int,
    learned_scales: Dict[str, dict] = None
) -> Dict[str, UnsignedUniformQuantizer]:
    """
    Calibrate quantizers for each layer.
    
    Args:
        activations: Dictionary of collected activations
        num_bits: Number of bits for quantization
        learned_scales: Optional dict of learned scales from QAT (alpha, levels per layer)
        
    Returns:
        Dictionary of calibrated quantizers per layer
    """
    quantizers = {}
    
    # Check if we have learned scales from QAT
    # IMPORTANT: Only use learned scales if the bit-width matches training!
    using_learned_scales = False
    if learned_scales is not None and len(learned_scales) > 0:
        # Check if any layer has the 'num_bits' field
        first_layer_scales = next(iter(learned_scales.values()))
        qat_num_bits = first_layer_scales.get('num_bits', None)
        
        if qat_num_bits == num_bits:
            using_learned_scales = True
            print(f"  Using learned quantization scales from QAT training ({qat_num_bits}-bit)!")
        else:
            print(f"  QAT trained with {qat_num_bits}-bit, but quantizing to {num_bits}-bit → using calibration instead")
    
    for layer_name, acts_list in activations.items():
        # Concatenate all activations for this layer
        all_acts = torch.cat([a.flatten() for a in acts_list])
        
        # Create unsigned quantizer (ReLU activations are always >= 0)
        quantizer = UnsignedUniformQuantizer(num_bits)
        
        # Use learned scale if available
        if using_learned_scales and layer_name in learned_scales:
            alpha = learned_scales[layer_name]['alpha']
            # LSQ: scale (alpha) is the step size between quantization levels
            # For unsigned: levels = [0, 1, 2, ..., Qp] * alpha
            # So: scale = alpha, max_val = Qp * alpha
            quantizer.scale = alpha
            quantizer.max_val = quantizer.qmax * alpha
            
            act_min = all_acts.min().item()
            act_max = all_acts.max().item()
            act_mean = all_acts.mean().item()
            act_std = all_acts.std().item()
            levels = learned_scales[layer_name]['levels']
            print(f"  {layer_name}: range=[{act_min:.4f}, {act_max:.4f}], mean={act_mean:.4f}, std={act_std:.4f}, learned_alpha={alpha:.4f}, levels={[f'{l:.3f}' for l in levels]}")
        else:
            # Standard calibration using max value
            quantizer.calibrate(all_acts)
            
            act_min = all_acts.min().item()
            act_max = all_acts.max().item()
            act_mean = all_acts.mean().item()
            act_std = all_acts.std().item()
            print(f"  {layer_name}: range=[{act_min:.4f}, {act_max:.4f}], mean={act_mean:.4f}, std={act_std:.4f}, max_val={quantizer.max_val:.4f}, scale={quantizer.scale:.4f}")
        
        quantizers[layer_name] = quantizer
    
    return quantizers


def compute_quantization_mse(
    activations: Dict[str, List[torch.Tensor]],
    quantizers: Dict[str, UnsignedUniformQuantizer]
) -> Dict[str, float]:
    """
    Compute MSE after quantization for each layer.
    
    Args:
        activations: Dictionary of collected activations
        quantizers: Dictionary of calibrated quantizers
        
    Returns:
        Dictionary of MSE values per layer
    """
    mse_results = {}
    
    for layer_name, acts_list in activations.items():
        quantizer = quantizers[layer_name]
        
        # Quantize and dequantize all activations
        all_mse = []
        for acts in acts_list:
            acts_q = quantizer.quantize_dequantize(acts)
            mse = torch.mean((acts - acts_q) ** 2).item()
            all_mse.append(mse)
        
        # Average MSE across batches
        mse_results[layer_name] = np.mean(all_mse)
    
    return mse_results


def evaluate_quantized_model(
    model: nn.Module,
    test_loader: DataLoader,
    quantizers: Dict[str, UnsignedUniformQuantizer],
    layer_names: List[str],
    device: torch.device
) -> float:
    """
    Evaluate model accuracy with quantized activations.
    
    This simulates inference with activation quantization by quantizing
    activations on-the-fly during forward pass.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        quantizers: Dictionary of calibrated quantizers per layer
        layer_names: List of layer names to quantize
        device: Device to run on
        
    Returns:
        Test accuracy (%)
    """
    model.eval()
    correct = 0
    total = 0
    
    # Register hooks to quantize activations
    handles = []
    
    def make_quantize_hook(name):
        def hook(module, input, output):
            # Quantize-dequantize the output
            quantizer = quantizers[name]
            return quantizer.quantize_dequantize(output)
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(make_quantize_hook(name))
            handles.append(handle)
    
    # Evaluate
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Quantize activations and compute MSE')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['baseline', 'akure'], required=True,
                        help='Type of model (baseline=ReLU, akure=ClippedReLU)')
    parser.add_argument('--clip-value', type=float, default=6.0,
                        help='Clip value used during training for akure model (default: 6.0)')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4, 6, 8],
                        help='Bit-widths to test (default: 2 4 6 8)')
    parser.add_argument('--calib-samples', type=int, default=500,
                        help='Number of calibration samples (default: 500)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output', type=str, default='quantization_results.json',
                        help='Output JSON file for results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("Activation Quantization Evaluation (PTQ)")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Bit-widths: {args.bits}")
    print(f"Calibration samples: {args.calib_samples}")
    print(f"Device: {device}\n")
    
    # Load checkpoint first to get hyperparameters
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Get clip_value from checkpoint if available (for BOTH baseline and akure)
    if 'clip_value' in checkpoint:
        clip_value = checkpoint['clip_value']
        print(f"Found clip_value={clip_value} in checkpoint")
    else:
        clip_value = args.clip_value if args.model_type == 'akure' else None
        if args.model_type == 'akure':
            print(f"Using clip_value={clip_value} from command line")
        else:
            print(f"No clip_value in checkpoint, using standard ReLU (clip_value=None)")
    
    # Get learned quantization scales if this was converted from QAT
    learned_scales = checkpoint.get('learned_scales', None)
    if learned_scales:
        print(f"Found learned quantization scales from QAT training (converted model)")
    
    # Create model with correct architecture
    if args.model_type == 'baseline':
        model = PlainConvFlatten(input_channels=1, num_classes=10, base=16, clip_value=clip_value)
        layer_names = ['relu1', 'relu2', 'relu3', 'relu4']
        if clip_value is not None:
            print(f"Baseline model with ClippedReLU (clip_value={clip_value})")
        else:
            print(f"Baseline model with standard ReLU")
    else:  # akure
        model = PlainConvFlattenReLU6(input_channels=1, num_classes=10, base=16, clip_value=clip_value)
        layer_names = ['relu1', 'relu2', 'relu3', 'relu4']
        print(f"A-KURE model with clip_value={clip_value}")
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded. Layers to quantize: {layer_names}\n")
    
    # Load calibration data
    print(f"Loading calibration data ({args.calib_samples} samples)...")
    # Use uint8 format to match training data format
    if args.model_type == 'akure':
        # A-KURE models expect uint8 [0, 255]
        transform = transforms.Lambda(lambda x: (transforms.ToTensor()(x) * 255).to(torch.uint8))
    else:
        # Baseline expects float [0, 1]
        transform = transforms.Compose([transforms.ToTensor()])
    
    full_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    
    # Subset for calibration
    calib_indices = np.random.choice(len(full_dataset), args.calib_samples, replace=False)
    calib_dataset = Subset(full_dataset, calib_indices)
    calib_loader = DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load test data for accuracy evaluation
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Collect activations
    print("Collecting activations from calibration data...")
    activations = collect_activations(model, calib_loader, device, layer_names)
    print(f"Collected activations from {len(layer_names)} layers.\n")
    
    # Evaluate FP32 accuracy (baseline)
    print("\nEvaluating FP32 baseline accuracy...")
    fp32_accuracy = evaluate_quantized_model(model, test_loader, {}, [], device)  # No quantization
    print(f"FP32 Accuracy: {fp32_accuracy:.2f}%\n")
    
    # Quantize at different bit-widths
    results = {}
    
    for num_bits in args.bits:
        print(f"\n{'='*70}")
        print(f"Quantizing at {num_bits} bits")
        print(f"{'='*70}")
        
        # Calibrate quantizers
        print("Calibrating quantizers...")
        quantizers = calibrate_quantizers(activations, num_bits, learned_scales)
        
        # Compute MSE
        print("\nComputing MSE...")
        mse_results = compute_quantization_mse(activations, quantizers)
        
        # Evaluate quantized accuracy
        print("\nEvaluating quantized model accuracy...")
        quant_accuracy = evaluate_quantized_model(model, test_loader, quantizers, layer_names, device)
        accuracy_drop = fp32_accuracy - quant_accuracy
        
        # Store quantization levels and activation statistics for each layer
        quant_levels = {}
        for layer_name, quantizer in quantizers.items():
            # Compute quantization levels
            levels = []
            for i in range(quantizer.qmax + 1):
                levels.append(i * quantizer.scale)
            
            # Get activation statistics
            acts_list = activations[layer_name]
            all_acts = torch.cat([a.flatten() for a in acts_list])
            
            quant_levels[layer_name] = {
                'scale': quantizer.scale,
                'max_val': quantizer.max_val,
                'levels': levels,
                'act_mean': all_acts.mean().item(),
                'act_std': all_acts.std().item(),
                'act_min': all_acts.min().item(),
                'act_max': all_acts.max().item()
            }
        
        # Store results
        results[f'{num_bits}bit'] = {
            'mse_per_layer': mse_results,
            'avg_mse': np.mean(list(mse_results.values())),
            'fp32_accuracy': fp32_accuracy,
            'accuracy': quant_accuracy,
            'accuracy_drop': accuracy_drop,
            'quantization_levels': quant_levels  # Save quantization levels!
        }
        
        # Print results
        print("\nMSE Results:")
        for layer_name, mse in mse_results.items():
            print(f"  {layer_name}: MSE = {mse:.6f}")
        print(f"  Average MSE: {results[f'{num_bits}bit']['avg_mse']:.6f}")
        print(f"\nAccuracy:")
        print(f"  FP32: {fp32_accuracy:.2f}%")
        print(f"  Quantized ({num_bits}-bit): {quant_accuracy:.2f}%")
        print(f"  Drop: {accuracy_drop:.2f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Summary
    print("\nSummary (Average MSE across layers):")
    print("-" * 40)
    for num_bits in args.bits:
        avg_mse = results[f'{num_bits}bit']['avg_mse']
        print(f"  {num_bits}-bit: {avg_mse:.6f}")


if __name__ == '__main__':
    main()

