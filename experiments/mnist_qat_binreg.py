"""
MNIST QAT with LSQ + Bin Regularization (BR)

This script implements proper Quantization-Aware Training (QAT) with:
1. LSQ (Learned Step-size Quantization) for activations with learnable step size (s)
2. Bin Regularization to encourage activations to cluster tightly at quantization levels
3. Two-stage training strategy (BR paper's S2 strategy):
   - Stage 1: Warmup (~30 epochs) - learn LSQ step size (s) from data (no BR)
   - Stage 2: Joint training - add BR loss while continuing to optimize s
   
Key Insight:
- LSQ learns WHERE the quantization grid should be (via learned step size s)
- BR makes activations STICK to that grid (minimize within-bin variance)
- They co-evolve: LSQ adjusts grid position, BR shapes activations to fit it
- The paper does NOT freeze s after warmup - it continues optimizing throughout QAT

Usage:
    python experiments/mnist_qat_binreg.py --num-bits 2 --warmup-epochs 30 --epochs 100 --lambda-br 0.1
"""

import os
import sys
import csv
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abr.hooks import ActivationHookManager
from abr.regularizer_binreg import BinRegularizer
from abr.lsq_quantizer import QuantizedClippedReLU


# ======================== Model Definition ========================

class PlainConvFlattenQAT(nn.Module):
    """
    Simple Conv model with QAT (Quantized Clipped ReLU activations).
    Activations are quantized during training with learnable scales.
    NO BatchNorm - it conflicts with bin regularization!
    """
    
    def __init__(self, input_channels=1, num_classes=10, base=16, clip_value=1.0, num_bits=2):
        super().__init__()
        
        self.clip_value = clip_value
        self.num_bits = num_bits
        
        # Downsample via strided conv (WITH bias now, since no BN)
        self.conv1 = nn.Conv2d(input_channels, base, 3, stride=2, padding=1, bias=True)
        self.relu1 = QuantizedClippedReLU(clip_value, num_bits)
        
        self.conv2 = nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=True)
        self.relu2 = QuantizedClippedReLU(clip_value, num_bits)
        
        self.conv3 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=True)
        self.relu3 = QuantizedClippedReLU(clip_value, num_bits)
        
        # Optional extra conv (no downsample)
        self.conv4 = nn.Conv2d(base*4, base*4, 3, stride=1, padding=1, bias=True)
        self.relu4 = QuantizedClippedReLU(clip_value, num_bits)
        
        # Head with Flatten
        # After 3 stride-2 convs, 28x28 -> 4x4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(base*4*4*4, 128, bias=True)
        self.relu5 = QuantizedClippedReLU(clip_value, num_bits)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Rescale from uint8 [0,255] to float [0,1]
        x = x.float() / 255.0
        
        # Conv blocks with quantized clipped ReLU (no BatchNorm!)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        # Head
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        
        return x


# ======================== Logging Functions ========================

def log_activation_histograms(writer, hook_manager, epoch, model=None):
    """
    Log activation histograms to TensorBoard.
    
    For QAT models, this logs:
    1. Pre-quantization activations (continuous, what BR shapes)
    2. Post-quantization activations (discrete, what flows to next layer)
    3. Quantization residuals (|pre - post|, shows BR effectiveness)
    4. ZOOMED versions focusing on quantization range for better comparison
    """
    # Get post-quantization activations (discrete)
    post_quant_activations = hook_manager.get_activations()
    
    # Get pre-quantization activations (continuous)
    pre_quant_activations = hook_manager.get_pre_quant_activations()
    
    # Get quantization modules to find max quantization level
    quant_modules = {}
    if model is not None:
        for name, module in model.named_modules():
            if isinstance(module, QuantizedClippedReLU):
                quant_modules[name] = module
    
    for name in post_quant_activations.keys():
        # Log post-quantization (discrete)
        post_acts_flat = post_quant_activations[name].flatten()
        writer.add_histogram(f'activations_post_quant/{name}', post_acts_flat, epoch)
        
        # Log pre-quantization (continuous) if available
        if name in pre_quant_activations:
            pre_acts_flat = pre_quant_activations[name].flatten()
            writer.add_histogram(f'activations_pre_quant/{name}', pre_acts_flat, epoch)
            
            # ZOOMED histograms: clip to quantization range [0, Qp*alpha]
            # This makes pre-quant and post-quant directly comparable!
            if name in quant_modules:
                levels = quant_modules[name].quantizer.get_quantization_levels()
                max_level = levels[-1].item()
                
                # Clip pre-quant to quantization range for fair comparison
                pre_acts_zoomed = torch.clamp(pre_acts_flat, 0, max_level)
                writer.add_histogram(f'activations_pre_quant_ZOOMED/{name}', pre_acts_zoomed, epoch)
                
                # Post-quant should already be in this range, but log for comparison
                writer.add_histogram(f'activations_post_quant_ZOOMED/{name}', post_acts_flat, epoch)
            
            # Log quantization residual: |pre - post|
            # This is THE KEY metric for BR effectiveness!
            # If BR is working, residuals should be very small (activations near levels)
            residual = torch.abs(pre_acts_flat - post_acts_flat)
            writer.add_histogram(f'quant_residual/{name}', residual, epoch)
            
            # Also log residual statistics
            writer.add_scalar(f'quant_residual_stats/{name}/mean', residual.mean().item(), epoch)
            writer.add_scalar(f'quant_residual_stats/{name}/max', residual.max().item(), epoch)
            writer.add_scalar(f'quant_residual_stats/{name}/std', residual.std().item(), epoch)


def log_quantization_scales(writer, model, epoch):
    """Log learned quantization scales (alpha)."""
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            alpha = module.quantizer.alpha.item()
            writer.add_scalar(f'quant_scales/{name}', alpha, epoch)
            
            # Also log quantization levels
            levels = module.quantizer.get_quantization_levels()
            for i, level in enumerate(levels):
                writer.add_scalar(f'quant_levels/{name}/level_{i}', level.item(), epoch)


def log_activation_clustering_plot(writer, model, hook_manager, epoch):
    """
    Create custom matplotlib plots showing pre-quant activations clustering around quantization levels.
    
    This creates a visual proof that BR is working: pre-quant activations should form
    peaks/clusters at the discrete quantization levels (shown as red dashed lines).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    pre_quant_activations = hook_manager.get_pre_quant_activations()
    post_quant_activations = hook_manager.get_activations()
    
    # Get quantization modules
    quant_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            quant_modules[name] = module
    
    for layer_name in pre_quant_activations.keys():
        if layer_name not in quant_modules:
            continue
            
        # Get data
        pre_acts = pre_quant_activations[layer_name].detach().cpu().flatten().numpy()
        post_acts = post_quant_activations[layer_name].detach().cpu().flatten().numpy()
        
        # Get quantization levels
        levels = quant_modules[layer_name].quantizer.get_quantization_levels().detach().cpu().numpy()
        alpha = quant_modules[layer_name].quantizer.alpha.item()
        Qp = quant_modules[layer_name].quantizer.Qp
        max_level = levels[-1]  # Qp * alpha
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot 1: Pre-quant activations FULL RANGE [0, clip_value]
        ax1.hist(pre_acts, bins=100, alpha=0.7, color='blue', edgecolor='black', density=True)
        ax1.set_title(f'{layer_name}: Pre-Quantization Activations - FULL RANGE [0, clip_value]\nValues > {max_level:.3f} will be clipped to level {Qp}', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Activation Value')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1.0])  # Full clip_value range
        
        # Draw vertical lines at quantization levels
        for i, level in enumerate(levels):
            ax1.axvline(level, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                       label=f'Level {i} = {level:.3f}')
        ax1.axvspan(max_level, 1.0, alpha=0.2, color='gray', label=f'Clipped to level {Qp}')
        ax1.legend(loc='upper right', fontsize=9)
        
        # Plot 2: Pre-quant activations ZOOMED to quantization range [0, Qp*alpha]
        ax2.hist(pre_acts, bins=100, alpha=0.7, color='blue', edgecolor='black', density=True, range=(0, max_level * 1.1))
        ax2.set_title(f'{layer_name}: Pre-Quantization - ZOOMED to Quantization Range [0, {max_level:.3f}]\nBR should create peaks at red lines!', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Activation Value')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, max_level * 1.1])  # Zoom to quantization range
        
        # Draw vertical lines at quantization levels on zoomed plot
        for i, level in enumerate(levels):
            ax2.axvline(level, color='red', linestyle='--', linewidth=3, alpha=0.9)
        
        # Add text annotation
        ax2.text(0.02, 0.95, f'Alpha (step size) = {alpha:.4f}\nNum levels = {len(levels)}\nQp = {Qp}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
        
        # Plot 3: Post-quant activations (should be discrete spikes)
        ax3.hist(post_acts, bins=100, alpha=0.7, color='green', edgecolor='black', density=True)
        ax3.set_title(f'{layer_name}: Post-Quantization Activations (Discrete)\nShould be sharp spikes ONLY at red lines', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Activation Value')
        ax3.set_ylabel('Density')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, max_level * 1.1])  # Same zoom as pre-quant
        
        # Draw vertical lines at quantization levels
        for level in levels:
            ax3.axvline(level, color='red', linestyle='--', linewidth=3, alpha=0.9)
        
        plt.tight_layout()
        
        # Log to TensorBoard
        writer.add_figure(f'clustering_plot/{layer_name}', fig, epoch)
        plt.close(fig)


def log_binreg_scalars(writer, regularizer, info_dict, epoch):
    """Log bin regularization metrics to TensorBoard."""
    writer.add_scalar('binreg/avg_loss', info_dict['avg_loss'], epoch)
    writer.add_scalar('binreg/br_mse_loss', info_dict['avg_mse'], epoch)  # BR internal loss component
    writer.add_scalar('binreg/br_var_loss', info_dict['avg_var'], epoch)  # BR internal loss component
    writer.add_scalar('binreg/quantization_mse', info_dict.get('avg_quantization_mse', 0), epoch)  # ACTUAL quant error!
    
    # NEW: BR Effectiveness Metrics
    writer.add_scalar('binreg/effectiveness', info_dict['avg_effectiveness'], epoch)
    writer.add_scalar('binreg/mean_distance', info_dict['avg_mean_distance'], epoch)
    writer.add_scalar('binreg/pct_near_levels', info_dict['avg_pct_near'], epoch)
    
    # Per-layer losses and effectiveness
    for layer_name, layer_info in info_dict['layer_losses'].items():
        writer.add_scalar(f'binreg_layer/{layer_name}/loss', layer_info['loss'], epoch)
        writer.add_scalar(f'binreg_layer/{layer_name}/mse', layer_info['mse'], epoch)
        writer.add_scalar(f'binreg_layer/{layer_name}/var', layer_info['var'], epoch)
        writer.add_scalar(f'binreg_layer/{layer_name}/effectiveness', layer_info['effectiveness'], epoch)
        writer.add_scalar(f'binreg_layer/{layer_name}/mean_distance', layer_info['mean_distance'], epoch)


def log_activation_statistics(writer, hook_manager, epoch):
    """Log activation statistics (mean, std, etc) to TensorBoard."""
    activations = hook_manager.get_activations()
    
    for name, acts in activations.items():
        acts_flat = acts.flatten()
        writer.add_scalar(f'activation_stats/{name}/mean', acts_flat.mean().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/std', acts_flat.std().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/min', acts_flat.min().item(), epoch)
        writer.add_scalar(f'activation_stats/{name}/max', acts_flat.max().item(), epoch)


# ======================== Helper Functions ========================

def get_layer_alphas(model, layer_names, detach=True):
    """
    Extract current alpha values from LSQ quantizers.
    
    BR MUST use these same alphas to compute quantization levels!
    This ensures BR and LSQ use the same targets.
    
    Args:
        model: The model containing LSQ quantizers
        layer_names: List of layer names to extract alphas from
        detach: If True, return Python floats (no gradient to alpha)
                If False, return tensors (allows BR to backprop into alpha)
                
    CRITICAL DESIGN CHOICE:
    - detach=True (default): BR optimizes activations given fixed grid.
                             LSQ optimizes grid separately. Decoupled.
    - detach=False (paper-faithful): BR can backprop into alpha/s.
                                     "Step sizes updated simultaneously" via combined loss.
                                     May be less stable early (hence warmup).
    
    Returns:
        Dict[layer_name, alpha_value] (float if detach=True, tensor if False)
    """
    alphas = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU) and name in layer_names:
            if detach:
                alphas[name] = module.quantizer.alpha.item()  # Python float, no gradient
            else:
                alphas[name] = module.quantizer.alpha.squeeze()  # Tensor, keeps gradient
    return alphas


# ======================== Training Functions ========================

def train_epoch(model, loader, optimizer, criterion, hook_manager, regularizer, lambda_br, device, use_br=True, br_backprop_to_alpha=False):
    """Train for one epoch."""
    model.train()
    hook_manager.set_training_mode(True)
    
    total_loss = 0
    total_task_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (with quantization!)
        output = model(data)
        
        # Task loss (L_CE)
        task_loss = criterion(output, target)
        
        # Bin regularization loss (L_BR) - only if enabled
        # CRITICAL: BR must operate on PRE-quantization activations (continuous)
        # NOT on post-quantization activations (discrete)
        if use_br:
            pre_quant_activations = hook_manager.get_pre_quant_activations()
            # Get current alpha values from LSQ quantizers (BR must use SAME levels as LSQ!)
            # detach=(not br_backprop_to_alpha): if backprop enabled, keep tensor with gradient
            alphas = get_layer_alphas(model, hook_manager.registered_layers, detach=(not br_backprop_to_alpha))
            br_loss, _ = regularizer.compute_total_loss(pre_quant_activations, alphas)
            total_loss_value = task_loss + lambda_br * br_loss
            reg_loss_value = (lambda_br * br_loss).item() if isinstance(br_loss, torch.Tensor) else (lambda_br * br_loss)
        else:
            total_loss_value = task_loss
            reg_loss_value = 0.0
        
        # Backward pass
        total_loss_value.backward()
        optimizer.step()
        
        # Metrics
        total_loss += total_loss_value.item()
        total_task_loss += task_loss.item()
        total_reg_loss += reg_loss_value
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return (total_loss / len(loader), 
            total_task_loss / len(loader),
            total_reg_loss / len(loader),
            100. * correct / total)


def test_epoch(model, loader, criterion, hook_manager, regularizer, lambda_br, device, use_br=True, br_backprop_to_alpha=False):
    """Evaluate on test set."""
    model.eval()
    hook_manager.set_training_mode(False)
    
    test_loss = 0
    test_task_loss = 0
    test_reg_loss = 0
    correct = 0
    total = 0
    
    # Accumulate BR metrics across all batches (FIX: was only keeping last batch!)
    accumulated_effectiveness = 0
    accumulated_mean_distance = 0
    accumulated_pct_near = 0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            # Task loss (L_CE)
            task_loss = criterion(output, target)
            
            # Bin regularization loss (L_BR) - only if enabled
            if use_br:
                activations = hook_manager.get_pre_quant_activations()  # FIX: Use pre-quant!
                # Get current alpha values from LSQ quantizers (BR must use SAME levels as LSQ!)
                # For test, always detach (no gradient needed)
                alphas = get_layer_alphas(model, hook_manager.registered_layers, detach=True)
                br_loss, info_dict = regularizer.compute_total_loss(activations, alphas)
                loss = task_loss + lambda_br * br_loss
                reg_loss_value = (lambda_br * br_loss).item() if isinstance(br_loss, torch.Tensor) else (lambda_br * br_loss)
                
                # Accumulate BR metrics
                accumulated_effectiveness += info_dict['avg_effectiveness']
                accumulated_mean_distance += info_dict['avg_mean_distance']
                accumulated_pct_near += info_dict['avg_pct_near']
                num_batches += 1
            else:
                loss = task_loss
                reg_loss_value = 0.0
            
            test_loss += loss.item()
            test_task_loss += task_loss.item()
            test_reg_loss += reg_loss_value
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    # Compute averaged BR metrics across all test batches
    if use_br and num_batches > 0:
        info_dict = {
            'avg_loss': test_reg_loss / len(loader),  # BR loss component
            'avg_effectiveness': accumulated_effectiveness / num_batches,
            'avg_mean_distance': accumulated_mean_distance / num_batches,
            'avg_pct_near': accumulated_pct_near / num_batches,
            'avg_quantization_mse': (accumulated_mean_distance / num_batches) ** 2,  # Estimate from mean distance
            'avg_mse': 0,  # BR internal loss component (not meaningful here)
            'avg_var': 0,  # BR internal loss component (not meaningful here)
            'layer_losses': {}
        }
    else:
        info_dict = {'avg_loss': 0, 'avg_effectiveness': 0, 'avg_mean_distance': 0, 'avg_pct_near': 0, 
                     'avg_quantization_mse': 0, 'avg_mse': 0, 'avg_var': 0, 'layer_losses': {}}
    
    return (test_loss / len(loader),
            test_task_loss / len(loader),
            test_reg_loss / len(loader),
            100. * correct / total,
            info_dict)


# ======================== Main Training Loop ========================

def main():
    parser = argparse.ArgumentParser(description='MNIST QAT with LSQ + Bin Regularization')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=30, help='Total number of epochs (default: 30)')
    parser.add_argument('--warmup-epochs', type=int, default=30, help='Warmup epochs (learn scales only) - BR paper uses 30 (default: 30)')
    parser.add_argument('--freeze-alpha', action='store_true', help='[EXPERIMENTAL] Freeze alpha after warmup. NOT recommended by BR paper - they optimize s throughout QAT.')
    parser.add_argument('--br-backprop-to-alpha', action='store_true',
                        help='[PAPER-FAITHFUL] Allow BR loss to backprop into alpha/s. '
                             'Paper says "step sizes updated simultaneously via combined loss". '
                             'Default (False) uses detached alpha - BR only affects activations.')
    parser.add_argument('--manual-uniform-levels', action='store_true', 
                        help='[WRONG! DO NOT USE] Force uniform levels and freeze alpha. '
                             'This defeats the purpose of LSQ (learned step size). '
                             'BR paper uses LSQ\'s data-driven learned levels, NOT fixed uniform levels.')
    parser.add_argument('--br-all-layers', action='store_true',
                        help='Apply BR to ALL layers (including relu1 and relu5). '
                             'Default: only middle layers (relu2, relu3, relu4)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--num-bits', type=int, default=2, help='Target bit-width for quantization (default: 2)')
    parser.add_argument('--clip-value', type=float, default=1.0, help='ReLU clip value (default: 1.0)')
    parser.add_argument('--lambda-br', type=float, default=0.1, help='Lambda for bin regularization loss (default: 0.1)')
    parser.add_argument('--pretrained-baseline', type=str, default=None,
                        help='Optional path to a baseline FP32 checkpoint (from experiments/mnist_baseline.py). '
                             'If provided, loads matching weights into the QAT model (strict=False) to warm-start QAT-BR. '
                             'This is the fairest way to compare QAT-BR vs Baseline+PTQ.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("MNIST QAT with LSQ + Bin Regularization")
    print("="*70)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Clip value (ReLU): {args.clip_value}")
    print(f"Target bit-width: {args.num_bits} bits ({2**args.num_bits} levels)")
    print(f"Lambda BR: {args.lambda_br}")
    print(f"Learning rate: {args.lr} (initial) with cosine annealing")
    print(f"Weight decay: 1e-4")
    print(f"Total epochs: {args.epochs}")
    
    if args.manual_uniform_levels:
        print(f"")
        print(f"⚠️  WARNING: MANUAL UNIFORM LEVELS MODE (NOT recommended by BR paper)")
        print(f"   - Alpha initialized to {args.clip_value}/{2**args.num_bits - 1} = uniform spacing")
        print(f"   - Alpha FROZEN (no LSQ adaptation)")
        print(f"   - This defeats LSQ's data-driven learning!")
        print(f"   - BR paper uses LEARNED levels from LSQ, not fixed uniform levels")
    else:
        print(f"  - Warmup: {args.warmup_epochs} epochs (learn alpha via LSQ only, no BR)")
        print(f"  - Fine-tune: {args.epochs - args.warmup_epochs} epochs (LSQ + BR jointly)")
        print(f"  - Alpha continues to be optimized throughout (as per BR paper S2 strategy)")
        if args.freeze_alpha:
            print(f"  ⚠️  ALPHA FREEZE ENABLED: alpha will be frozen after warmup")
            print(f"      (EXPERIMENTAL - NOT recommended by BR paper, which optimizes s throughout QAT)")
    
    # BR backprop mode
    print(f"")
    if args.br_backprop_to_alpha:
        print(f"  🔬 BR BACKPROP TO ALPHA: ENABLED (paper-faithful)")
        print(f"     - BR loss can backprop into alpha/s")
        print(f"     - 'Step sizes updated simultaneously via combined loss' (paper)")
        print(f"     - May be less stable early (warmup should help)")
    else:
        print(f"  🔬 BR BACKPROP TO ALPHA: DISABLED (default, decoupled)")
        print(f"     - BR loss only affects activations (alpha detached)")
        print(f"     - LSQ updates alpha separately via quantization loss")
        print(f"     - More stable but may leave performance on table")
        print(f"     - This is a VARIANT of the paper's approach")
    print("="*70)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    log_dir = f'./runs/mnist_qat_binreg_{timestamp}'
    checkpoint_path = f'./checkpoints/mnist_qat_binreg_{timestamp}.pth'
    csv_path = f'./logs/mnist_qat_binreg_{timestamp}.csv'
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Data (uint8 format [0, 255])
    transform = transforms.Lambda(lambda x: (transforms.ToTensor()(x) * 255).to(torch.uint8))
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model with QAT
    model = PlainConvFlattenQAT(
        input_channels=1, 
        num_classes=10, 
        base=16, 
        clip_value=args.clip_value,
        num_bits=args.num_bits
    ).to(device)

    # Optional: warm-start from baseline FP32 weights
    if args.pretrained_baseline is not None:
        print("\n" + "="*70)
        print("WARM-START: LOADING BASELINE FP32 WEIGHTS")
        print("="*70)
        print(f"Loading from: {args.pretrained_baseline}")
        base_ckpt = torch.load(args.pretrained_baseline, map_location=device)
        base_sd = base_ckpt['model_state_dict'] if isinstance(base_ckpt, dict) and 'model_state_dict' in base_ckpt else base_ckpt
        missing, unexpected = model.load_state_dict(base_sd, strict=False)
        print(f"✓ Loaded baseline weights into QAT model (strict=False)")
        if missing:
            # Expected: QAT quantizer params like *.quantizer.alpha, *.quantizer.init_state
            print(f"  Missing keys (expected for QAT-only params): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (baseline-only): {len(unexpected)}")
        print("="*70)
    
    # MANUAL UNIFORM LEVEL INITIALIZATION (if requested)
    # This overrides LSQ's data-driven initialization with uniform spacing
    # Result: alpha = clip_value / Qp, giving evenly-spaced levels across [0, clip_value]
    # These levels match what a uniform quantizer would use!
    if args.manual_uniform_levels:
        print("\n" + "="*70)
        print("MANUAL UNIFORM LEVEL INITIALIZATION")
        print("="*70)
        Qp = 2**args.num_bits - 1
        alpha_uniform = args.clip_value / Qp
        print(f"Setting alpha = {args.clip_value} / {Qp} = {alpha_uniform:.6f}")
        print(f"This gives uniform levels spanning [0, {args.clip_value}]")
        print(f"Alpha will be FROZEN (prevents LSQ from escaping)")
        print(f"")
        
        for name, module in model.named_modules():
            if isinstance(module, QuantizedClippedReLU):
                # Set alpha to uniform spacing
                module.quantizer.alpha.data.fill_(alpha_uniform)
                # Mark as initialized (prevent LSQ's data-driven init from overwriting)
                module.quantizer.init_state.fill_(1)
                # Freeze it (no gradient updates)
                module.quantizer.alpha.requires_grad = False
                print(f"  {name}: alpha={alpha_uniform:.6f} (FROZEN)")
        print("="*70)
        print(f"✓ BR will push activations toward these FIXED uniform levels")
        print(f"✓ No warmup needed (alpha already optimal for uniform quantization)")
        print("="*70 + "\n")
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Clip value: {args.clip_value}")
    print(f"Quantization bits: {args.num_bits}")
    
    # Print initial scales (after potential manual override)
    if not args.manual_uniform_levels:
        print(f"\nInitial quantization scales (data-driven from LSQ):")
        for name, module in model.named_modules():
            if isinstance(module, QuantizedClippedReLU):
                print(f"  {name}: alpha={module.quantizer.alpha.item():.4f}, levels={module.quantizer.get_quantization_levels().tolist()}")
    
    # Hook manager (register hooks on quantized ReLU activations)
    # Determine which layers to apply BR to
    if args.br_all_layers:
        br_layers = ['relu1', 'relu2', 'relu3', 'relu4', 'relu5']  # All layers
        print("BR will be applied to ALL layers (relu1-relu5)")
    else:
        br_layers = ['relu2', 'relu3', 'relu4']  # Only middle layers (default)
        print("BR will be applied to middle layers only (relu2-relu4)")
    
    # Hook manager for BR
    # IMPORTANT: 
    # 1. detach_activations=False to allow gradients for regularization
    # 2. BR loss is computed on PRE-quantization activations via get_pre_quant_activations()
    #    These are the continuous values after ReLU/clip, before LSQ round_pass
    # 3. Post-quantization activations (discrete) are available via get_activations()
    hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],
        layer_names=br_layers,
        exclude_first_last=False,
        detach_activations=False  # Keep gradients for regularization
    )
    
    # Separate hook manager for visualization (all layers)
    viz_hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],
        layer_names=['relu1', 'relu2', 'relu3', 'relu4', 'relu5'],  # ALL layers for histograms!
        exclude_first_last=False,
        detach_activations=True  # No gradients needed for visualization
    )
    print(f"\nHooked layers: {hook_manager.layer_names}")
    
    # Bin regularizer (uses quantization levels from model)
    regularizer = BinRegularizer(
        num_bits=args.num_bits
    )
    
    # Lambda for weighting BR loss (paper formulation: L = L_CE + λ · L_BR)
    lambda_br = args.lambda_br
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler: ReduceLROnPlateau
    # Reduces LR when validation accuracy plateaus (more adaptive than cosine annealing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # Maximize test accuracy
        factor=0.5,           # Reduce LR by half
        patience=10,          # Wait 10 epochs before reducing
        min_lr=1e-6           # Don't go below this
    )
    print(f"  Learning rate scheduler: ReduceLROnPlateau (patience={10}, factor={0.5})")
    
    print("\n" + "="*70)
    print("Starting Two-Stage Training (BR Paper S2 Strategy)")
    print("="*70)
    print("Strategy:")
    print("  1. Stage 1 (Warmup): LSQ learns optimal step size (s/alpha) from data")
    print("     - No BR loss, only task loss + quantization")
    print("     - Lets s stabilize to data-driven optimal values")
    print("  2. Stage 2 (Joint): Add BR to push activations toward LSQ's learned levels")
    print("     - BR targets = [0, s, 2s, 3s, ..., Qp·s] (from LSQ's learned s)")
    print("     - LSQ continues to optimize s throughout (NOT frozen)")
    print("     - BR makes activations 'stick' to the grid that LSQ defines")
    print("="*70)
    
    # Verify manual uniform levels are preserved (if used)
    if args.manual_uniform_levels:
        print("\nVerifying manual uniform levels before training:")
        for name, module in model.named_modules():
            if isinstance(module, QuantizedClippedReLU) and name in br_layers:
                alpha_val = module.quantizer.alpha.item()
                expected = args.clip_value / (2**args.num_bits - 1)
                if abs(alpha_val - expected) < 0.0001:
                    print(f"  ✓ {name}: alpha={alpha_val:.6f} (correct)")
                else:
                    print(f"  ❌ {name}: alpha={alpha_val:.6f} (WRONG! Expected {expected:.6f})")
        print()
    
    # Training loop with two stages
    for epoch in range(args.epochs):
        # Determine which stage we're in
        if args.manual_uniform_levels:
            # With manual uniform levels, alpha is already correct and frozen
            # No warmup needed - BR active from epoch 0
            is_warmup = False
            use_br = True
            stage_name = "BR TRAINING (manual uniform levels)"
        else:
            # Normal two-stage: warmup scales, then BR
            is_warmup = (epoch < args.warmup_epochs)
            use_br = not is_warmup
            stage_name = "WARMUP (scales only)" if is_warmup else "FINE-TUNE (scales + BR)"
        
        # Freeze alpha after warmup if requested (forces BR to work with fixed levels)
        if args.freeze_alpha and epoch == args.warmup_epochs:
            print("\n" + "="*70)
            print("FREEZING ALPHA (preventing LSQ from escaping BR)")
            print("="*70)
            for name, module in model.named_modules():
                if isinstance(module, QuantizedClippedReLU):
                    module.quantizer.alpha.requires_grad = False
                    alpha_val = module.quantizer.alpha.item()
                    print(f"  {name}: alpha={alpha_val:.4f} (FROZEN)")
            print("="*70 + "\n")
        
        # Train
        train_loss, train_task_loss, train_reg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, hook_manager, regularizer, lambda_br, device, 
            use_br=use_br, br_backprop_to_alpha=args.br_backprop_to_alpha
        )
        
        # Test
        test_loss, test_task_loss, test_reg_loss, test_acc, info_dict = test_epoch(
            model, test_loader, criterion, hook_manager, regularizer, lambda_br, device, 
            use_br=use_br, br_backprop_to_alpha=args.br_backprop_to_alpha
        )
        
        # Log to TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('loss/train_task', train_task_loss, epoch)
        writer.add_scalar('loss/test_task', test_task_loss, epoch)
        writer.add_scalar('loss/train_reg', train_reg_loss, epoch)
        writer.add_scalar('loss/test_reg', test_reg_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/test', test_acc, epoch)
        writer.add_scalar('training/stage', 0 if is_warmup else 1, epoch)
        
        # Log quantization scales
        log_quantization_scales(writer, model, epoch)
        
        # Log BR metrics (only if using BR)
        if use_br:
            log_binreg_scalars(writer, regularizer, info_dict, epoch)
        
        log_activation_statistics(writer, hook_manager, epoch)
        
        # Log histograms periodically (use viz_hook_manager to get all layers)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            log_activation_histograms(writer, viz_hook_manager, epoch, model=model)
        
        # Log custom clustering plots ONLY at key checkpoints (matplotlib is slow)
        # - After warmup (when BR is about to start)
        # - At the final epoch (to show final clustering)
        if epoch == args.warmup_epochs or epoch == args.epochs - 1:
            log_activation_clustering_plot(writer, model, viz_hook_manager, epoch)
        
        # Flush TensorBoard writer to disk
        writer.flush()
        
        # Terminal output
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} [{stage_name}] (LR={current_lr:.6f}):")
        print(f"  Train - Loss: {train_loss:.4f}, Task: {train_task_loss:.4f}, Reg: {train_reg_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Task: {test_task_loss:.4f}, Reg: {test_reg_loss:.4f}, Acc: {test_acc:.2f}%")
        if use_br:
            # New informative metrics!
            print(f"  BR Effectiveness: {info_dict['avg_effectiveness']:.1f}% "
                  f"(MeanDist={info_dict['avg_mean_distance']:.4f}, "
                  f"@Levels={info_dict['avg_pct_near']:.1f}%)")
        print()
        
        # Step learning rate scheduler (ReduceLROnPlateau needs the metric)
        scheduler.step(test_acc)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', current_lr, epoch)
    
    # Save final model with hyperparameters and BR metrics
    checkpoint_data = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_acc,  # Renamed for clarity
        'clip_value': args.clip_value,
        'num_bits': args.num_bits,
        'lambda_br': args.lambda_br,
        'warmup_epochs': args.warmup_epochs,
        'freeze_alpha': args.freeze_alpha,
        'manual_uniform_levels': args.manual_uniform_levels,
    }
    
    # Add BR metrics if available
    if use_br and info_dict:
        checkpoint_data['br_effectiveness'] = info_dict['avg_effectiveness']
        checkpoint_data['br_mean_distance'] = info_dict['avg_mean_distance']
        checkpoint_data['br_pct_at_levels'] = info_dict['avg_pct_near']
    
    torch.save(checkpoint_data, checkpoint_path)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")
    
    # Final results
    print("\n" + "="*70)
    print("Final Results:")
    print(f"  Test Acc: {test_acc:.2f}%")
    if use_br:
        print(f"\n  BR Effectiveness: {info_dict['avg_effectiveness']:.1f}%")
        print(f"    - Mean Distance to Nearest Level: {info_dict['avg_mean_distance']:.6f}")
        print(f"    - % Activations @ Levels: {info_dict['avg_pct_near']:.1f}%")
        print(f"    - Quantization MSE: {info_dict.get('avg_quantization_mse', 0):.8f}  ← Actual error!")
        print(f"\n  Interpretation:")
        eff = info_dict['avg_effectiveness']
        qmse = info_dict.get('avg_quantization_mse', 0)
        if eff > 85:
            print(f"    ✓ Excellent clustering! Activations are tightly binned.")
            print(f"    ✓ Quantization MSE = {qmse:.8f} (near-perfect!)")
        elif eff > 70:
            print(f"    ✓ Good clustering. Consider higher λ for tighter bins.")
            print(f"    → Quantization MSE = {qmse:.8f} (some error remains)")
        elif eff > 50:
            print(f"    ⚠ Moderate clustering. Increase λ significantly.")
            print(f"    → Quantization MSE = {qmse:.8f} (noticeable error)")
        else:
            print(f"    ✗ Poor clustering. BR is not working effectively.")
    print("="*70)
    
    writer.close()


if __name__ == '__main__':
    main()

