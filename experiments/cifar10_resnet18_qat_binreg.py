#!/usr/bin/env python3
"""
CIFAR-10 ResNet18 with QAT + Bin Regularization

Fine-tune ResNet18 on CIFAR-10 with:
- LSQ (Learned Step-size Quantization) for activations
- Bin Regularization to cluster activations at quantization levels
- 2-bit quantization
- ClippedReLU (ReLU6 by default, configurable)

Usage:
    python experiments/cifar10_resnet18_qat_binreg.py \
        --pretrained checkpoints/cifar10_resnet18_baseline_XXX.pth \
        --num-bits 2 --clip-value 6.0 --lambda-br 0.5 \
        --warmup-epochs 5 --epochs 30 --gpu 0
    
    # Resume from checkpoint:
    python experiments/cifar10_resnet18_qat_binreg.py \
        --resume checkpoints/cifar10_resnet18_qat_binreg_XXX_latest.pth \
        --epochs 50 --gpu 0
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm

from abr.lsq_quantizer import QuantizedClippedReLU
from abr.regularizer_binreg import BinRegularizer
from abr.hooks import ActivationHookManager


def replace_relu_with_qrelu(module, clip_value, num_bits):
    """
    Recursively replace all ReLU layers with QuantizedClippedReLU.
    
    Args:
        module: PyTorch module to modify
        clip_value: Clip value for QuantizedClippedReLU
        num_bits: Number of bits for quantization
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6)):
            # Replace ReLU/ReLU6 with QuantizedClippedReLU
            setattr(module, name, QuantizedClippedReLU(clip_value, num_bits))
        else:
            # Recursively apply to children
            replace_relu_with_qrelu(child, clip_value, num_bits)


def get_resnet18_cifar10_qat(num_classes=10, pretrained_imagenet=True, clip_value=6.0, num_bits=2, pretrained_baseline=None):
    """
    Get ResNet18 with QAT for CIFAR-10.
    
    Args:
        num_classes: Number of output classes
        pretrained_imagenet: Whether to use ImageNet pretrained weights
        clip_value: Clip value for quantized ReLU
        num_bits: Number of bits for quantization
        pretrained_baseline: Path to baseline checkpoint (optional, takes priority over pretrained_imagenet)
    
    Returns:
        ResNet18 with QuantizedClippedReLU layers
    """
    # Load ResNet18
    if pretrained_imagenet and pretrained_baseline is None:
        print("  Loading ImageNet pretrained weights...")
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
    
    # Adapt for CIFAR-10 (32x32 images)
    # Replace 7x7 stride=2 conv with 3x3 stride=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool (too aggressive for 32x32)
    model.maxpool = nn.Identity()
    # Replace final FC for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Replace all ReLU with QuantizedClippedReLU
    replace_relu_with_qrelu(model, clip_value, num_bits)
    
    # Load baseline weights if provided
    if pretrained_baseline:
        print(f"  Loading pretrained baseline from: {pretrained_baseline}")
        checkpoint = torch.load(pretrained_baseline, map_location='cpu')
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
        
        # Filter out keys that don't match (quantizer parameters)
        model_dict = model.state_dict()
        
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if 'quantizer' not in k and k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"    Skipping {k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"  ✓ Loaded {len(filtered_dict)} parameter tensors from baseline")
    
    return model


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """Get CIFAR-10 train and test loaders."""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def log_activation_histograms(writer, hook_manager, epoch, prefix='pre_quant'):
    """Log activation histograms to TensorBoard."""
    activations = hook_manager.get_pre_quant_activations()
    
    for name, acts in activations.items():
        if acts is None or len(acts) == 0:
            continue
        
        all_acts = acts.flatten()
        writer.add_histogram(f'{prefix}/{name}', all_acts, epoch)
        
        # Statistics
        mean_val = all_acts.mean().item()
        std_val = all_acts.std().item()
        max_val = all_acts.max().item()
        min_val = all_acts.min().item()
        
        writer.add_scalar(f'{prefix}_stats/{name}/mean', mean_val, epoch)
        writer.add_scalar(f'{prefix}_stats/{name}/std', std_val, epoch)
        writer.add_scalar(f'{prefix}_stats/{name}/max', max_val, epoch)
        writer.add_scalar(f'{prefix}_stats/{name}/min', min_val, epoch)


def log_quantization_scales(writer, model, epoch):
    """Log learned LSQ alpha values."""
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            alpha = module.quantizer.alpha.item()
            writer.add_scalar(f'quantization/alpha/{name}', alpha, epoch)


def train_epoch(model, loader, optimizer, criterion, hook_manager, regularizer, lambda_br, device, use_br=True, br_backprop_to_alpha=False):
    """Train for one epoch with optional BR."""
    model.train()
    hook_manager.set_training_mode(True)
    
    total_loss = 0
    total_task_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        task_loss = criterion(output, target)
        
        # Compute BR loss if enabled
        if use_br:
            activations = hook_manager.get_pre_quant_activations()
            
            # Extract alphas from model
            # detach=(not br_backprop_to_alpha): if backprop enabled, keep tensor with gradient
            alphas = {}
            for name, module in model.named_modules():
                if isinstance(module, QuantizedClippedReLU):
                    if br_backprop_to_alpha:
                        alphas[name] = module.quantizer.alpha.squeeze()  # Tensor, keeps gradient
                    else:
                        alphas[name] = module.quantizer.alpha.item()  # Python float, no gradient
            
            br_loss, info_dict = regularizer.compute_total_loss(activations, alphas)
            loss = task_loss + lambda_br * br_loss
            reg_loss_val = br_loss if isinstance(br_loss, float) else br_loss.item()
        else:
            loss = task_loss
            reg_loss_val = 0.0
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_reg_loss += reg_loss_val
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    avg_loss = total_loss / len(loader)
    avg_task_loss = total_task_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_task_loss, avg_reg_loss, accuracy


def test_epoch(model, loader, criterion, hook_manager, regularizer, lambda_br, device, use_br=True):
    """Test for one epoch with optional BR metrics."""
    model.eval()
    hook_manager.set_training_mode(False)
    
    total_loss = 0
    total_task_loss = 0
    total_reg_loss = 0
    correct = 0
    total = 0
    
    # BR metrics accumulators
    if use_br:
        total_effectiveness = 0
        total_mean_distance = 0
        total_pct_near = 0
        total_quant_mse = 0
        num_br_batches = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            task_loss = criterion(output, target)
            
            # Compute BR metrics if enabled
            if use_br:
                activations = hook_manager.get_pre_quant_activations()
                
                # Extract alphas
                alphas = {}
                for name, module in model.named_modules():
                    if isinstance(module, QuantizedClippedReLU):
                        alphas[name] = module.quantizer.alpha.item()
                
                br_loss, info_dict = regularizer.compute_total_loss(activations, alphas)
                loss = task_loss + lambda_br * br_loss
                reg_loss_val = br_loss if isinstance(br_loss, float) else br_loss.item()
                
                # Accumulate BR metrics
                total_effectiveness += info_dict.get('avg_effectiveness', 0)
                total_mean_distance += info_dict.get('avg_mean_distance', 0)
                total_pct_near += info_dict.get('avg_pct_near', 0)
                total_quant_mse += info_dict.get('avg_quantization_mse', 0)
                num_br_batches += 1
            else:
                loss = task_loss
                reg_loss_val = 0.0
            
            # Statistics
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_reg_loss += reg_loss_val
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(loader)
    avg_task_loss = total_task_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    accuracy = 100. * correct / total
    
    # Average BR metrics
    info_dict = {}
    if use_br and num_br_batches > 0:
        info_dict = {
            'avg_effectiveness': total_effectiveness / num_br_batches,
            'avg_mean_distance': total_mean_distance / num_br_batches,
            'avg_pct_near': total_pct_near / num_br_batches,
            'avg_quantization_mse': total_quant_mse / num_br_batches,
        }
    else:
        info_dict = {
            'avg_effectiveness': 0,
            'avg_mean_distance': 0,
            'avg_pct_near': 0,
            'avg_quantization_mse': 0,
        }
    
    return avg_loss, avg_task_loss, avg_reg_loss, accuracy, info_dict


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 ResNet18 QAT + Bin Regularization')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained baseline model')
    parser.add_argument('--pretrained-imagenet', action='store_true', help='Use ImageNet pretrained weights (if no --pretrained baseline)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, help='Total epochs (default: 30)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs (default: 5)')
    parser.add_argument('--freeze-alpha', action='store_true', help='[EXPERIMENTAL] Freeze alpha after warmup. NOT recommended by BR paper.')
    parser.add_argument('--br-backprop-to-alpha', action='store_true',
                        help='[PAPER-FAITHFUL] Allow BR loss to backprop into alpha/s. '
                             'Paper says "step sizes updated simultaneously via combined loss". '
                             'Default (False) uses detached alpha - BR only affects activations.')
    parser.add_argument('--br-sample-layers', type=int, default=-1, help='Number of layers to apply BR to (default: -1 for all)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--num-bits', type=int, default=2, help='Quantization bits (default: 2)')
    parser.add_argument('--clip-value', type=float, default=6.0, help='ReLU clip value (default: 6.0)')
    parser.add_argument('--lambda-br', type=float, default=0.5, help='BR lambda (default: 0.5)')
    parser.add_argument('--manual-uniform-levels', action='store_true', 
                        help='[WRONG! DO NOT USE] Force uniform levels and freeze alpha. '
                             'This defeats the purpose of LSQ (learned step size).')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("CIFAR-10 ResNet18 with QAT + Bin Regularization")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Quantization bits: {args.num_bits}")
    print(f"Clip value: {args.clip_value}")
    print(f"Lambda BR: {args.lambda_br}")
    print(f"Manual uniform levels: {args.manual_uniform_levels}")
    print(f"Freeze alpha after warmup: {args.freeze_alpha}")
    print(f"BR backprop to alpha: {args.br_backprop_to_alpha}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Total epochs: {args.epochs}")
    print("="*70)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    resume_checkpoint = None
    
    if args.resume:
        print("\n" + "="*70)
        print("RESUMING FROM CHECKPOINT")
        print("="*70)
        print(f"Loading checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device)
        
        # Extract saved hyperparameters
        saved_epoch = resume_checkpoint.get('epoch', 0)
        start_epoch = saved_epoch + 1
        best_acc = resume_checkpoint.get('best_accuracy', resume_checkpoint.get('test_accuracy', 0.0))
        
        # Use saved hyperparameters (override args)
        args.clip_value = resume_checkpoint.get('clip_value', args.clip_value)
        args.num_bits = resume_checkpoint.get('num_bits', args.num_bits)
        args.lambda_br = resume_checkpoint.get('lambda_br', args.lambda_br)
        args.warmup_epochs = resume_checkpoint.get('warmup_epochs', args.warmup_epochs)
        args.freeze_alpha = resume_checkpoint.get('freeze_alpha', args.freeze_alpha)
        args.br_backprop_to_alpha = resume_checkpoint.get('br_backprop_to_alpha', args.br_backprop_to_alpha)
        args.manual_uniform_levels = resume_checkpoint.get('manual_uniform_levels', args.manual_uniform_levels)
        use_imagenet = resume_checkpoint.get('pretrained_imagenet', False)
        
        print(f"  Resuming from epoch {start_epoch} (saved epoch: {saved_epoch})")
        print(f"  Best accuracy so far: {best_acc:.2f}%")
        print(f"  Using saved hyperparameters:")
        print(f"    clip_value={args.clip_value}, num_bits={args.num_bits}")
        print(f"    lambda_br={args.lambda_br}, warmup_epochs={args.warmup_epochs}")
        print("="*70 + "\n")
    else:
        use_imagenet = args.pretrained_imagenet and args.pretrained is None
    
    # Create timestamp (use checkpoint timestamp if resuming, otherwise new)
    if args.resume:
        # Try to extract timestamp from checkpoint path or use current
        import os
        checkpoint_name = os.path.basename(args.resume)
        if 'qat_binreg_' in checkpoint_name:
            # Extract timestamp from filename like "cifar10_resnet18_qat_binreg_20260106_171947.pth"
            parts = checkpoint_name.split('_')
            if len(parts) >= 5:
                timestamp = f"{parts[-2]}_{parts[-1].replace('.pth', '')}"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    log_dir = f'./runs/cifar10_resnet18_qat_binreg_{timestamp}'
    checkpoint_path = f'./checkpoints/cifar10_resnet18_qat_binreg_{timestamp}.pth'
    
    # If resuming, use same checkpoint path (overwrite existing checkpoint)
    if args.resume:
        checkpoint_path = args.resume
        print(f"  Will save to same checkpoint: {checkpoint_path}")
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    
    # Model with QAT
    model = get_resnet18_cifar10_qat(
        num_classes=10,
        pretrained_imagenet=use_imagenet if not args.resume else False,
        clip_value=args.clip_value,
        num_bits=args.num_bits,
        pretrained_baseline=args.pretrained if not args.resume else None
    ).to(device)
    
    print(f"\nModel: ResNet18_QAT (adapted for CIFAR-10 + QuantizedClippedReLU)")
    if args.resume:
        print(f"  ✓ Resuming from checkpoint: {args.resume}")
    elif args.pretrained:
        print(f"  ✓ Loaded baseline weights from: {args.pretrained}")
    elif use_imagenet:
        print(f"  ✓ Started from ImageNet pretrained weights")
    else:
        print(f"  ✓ Training from scratch")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Count QuantizedClippedReLU layers
    qrelu_layers = [name for name, m in model.named_modules() if isinstance(m, QuantizedClippedReLU)]
    print(f"Total QuantizedClippedReLU layers: {len(qrelu_layers)}")
    
    # Show layer names
    print(f"\nQuantizedClippedReLU layers:")
    for i, name in enumerate(qrelu_layers, 1):
        print(f"  [{i}] {name}")
    print()
    
    # Determine which layers to apply BR to
    if args.br_sample_layers > 0 and args.br_sample_layers < len(qrelu_layers):
        # Sample evenly spaced layers
        step = len(qrelu_layers) // args.br_sample_layers
        br_layer_names = [qrelu_layers[i*step] for i in range(args.br_sample_layers)]
        print(f"✓ Applying BR to {len(br_layer_names)} sampled layers (every {step}th layer)")
    else:
        br_layer_names = qrelu_layers
        print(f"✓ Applying BR to ALL {len(br_layer_names)} layers")
    
    # Manual uniform level initialization
    if args.manual_uniform_levels:
        print("\n" + "="*70)
        print("MANUAL UNIFORM LEVEL INITIALIZATION")
        print("="*70)
        Qp = 2**args.num_bits - 1
        alpha_uniform = args.clip_value / Qp
        print(f"Setting alpha = {args.clip_value} / {Qp} = {alpha_uniform:.6f}")
        print(f"Alpha will be FROZEN")
        print("="*70)
        
        for name, module in model.named_modules():
            if isinstance(module, QuantizedClippedReLU):
                module.quantizer.alpha.data.fill_(alpha_uniform)
                module.quantizer.init_state.fill_(1)
                module.quantizer.alpha.requires_grad = False
        
        print("✓ All alpha values set to uniform spacing and frozen")
        print("="*70 + "\n")
    
    # Hook manager for BR
    hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],
        layer_names=br_layer_names,
        exclude_first_last=False,
        detach_activations=False
    )
    
    print(f"✓ Hooked {len(hook_manager.registered_layers)} layers for BR:")
    for i, layer in enumerate(hook_manager.registered_layers, 1):
        print(f"  [{i}] {layer}")
    print()
    
    # Bin regularizer
    regularizer = BinRegularizer(num_bits=args.num_bits)
    
    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    finetune_epochs = args.epochs - args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=args.lr * 0.01
    )
    
    # Load checkpoint state if resuming
    if args.resume and resume_checkpoint:
        print("\n" + "="*70)
        print("LOADING CHECKPOINT STATE")
        print("="*70)
        
        # Load model state
        if 'model_state_dict' in resume_checkpoint:
            model.load_state_dict(resume_checkpoint['model_state_dict'])
            print("  ✓ Loaded model state")
        else:
            print("  ⚠ Warning: No model_state_dict in checkpoint, using current model")
        
        # Load optimizer state
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            print("  ✓ Loaded optimizer state")
        else:
            print("  ⚠ Warning: No optimizer_state_dict in checkpoint")
        
        # Load scheduler state (if saved)
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            print("  ✓ Loaded scheduler state")
        else:
            # Manually step scheduler to correct position
            # Step scheduler to match the epoch we're resuming from
            for _ in range(max(0, start_epoch - args.warmup_epochs)):
                scheduler.step()
            print(f"  ✓ Scheduler stepped to epoch {start_epoch} position")
        
        # Restore alpha freeze state if applicable
        if resume_checkpoint.get('freeze_alpha', False):
            if start_epoch > resume_checkpoint.get('warmup_epochs', 0):
                print("  ✓ Restoring alpha freeze state...")
                for name, module in model.named_modules():
                    if isinstance(module, QuantizedClippedReLU):
                        module.quantizer.alpha.requires_grad = False
        
        print("="*70 + "\n")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    if args.resume:
        print("="*70)
        print(f"RESUMING TRAINING from epoch {start_epoch}/{args.epochs}")
        print("="*70)
    else:
        print("="*70)
        print("Starting Two-Stage Training...")
        print("="*70)
        print("Stage 1: Warmup - Learn LSQ scales only (no BR)")
        print("Stage 2: BR Training - Add bin regularization")
        print("="*70)
    
    for epoch in range(start_epoch, args.epochs):
        # Determine stage
        if args.manual_uniform_levels:
            is_warmup = False
            use_br = True
            stage_name = "BR TRAINING"
        else:
            is_warmup = (epoch < args.warmup_epochs)
            use_br = not is_warmup
            stage_name = "WARMUP" if is_warmup else "BR TRAINING"
        
        # Freeze alpha after warmup if requested
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
            model, train_loader, optimizer, criterion, hook_manager, regularizer, args.lambda_br, device, 
            use_br=use_br, br_backprop_to_alpha=args.br_backprop_to_alpha
        )
        
        # Test
        test_loss, test_task_loss, test_reg_loss, test_acc, info_dict = test_epoch(
            model, test_loader, criterion, hook_manager, regularizer, args.lambda_br, device, use_br=use_br
        )
        
        # Step scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', test_loss, epoch)
        writer.add_scalar('loss/train_task', train_task_loss, epoch)
        writer.add_scalar('loss/test_task', test_task_loss, epoch)
        writer.add_scalar('loss/train_reg', train_reg_loss, epoch)
        writer.add_scalar('loss/test_reg', test_reg_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/test', test_acc, epoch)
        writer.add_scalar('lr', current_lr, epoch)
        
        if use_br:
            writer.add_scalar('br/effectiveness', info_dict['avg_effectiveness'], epoch)
            writer.add_scalar('br/mean_distance', info_dict['avg_mean_distance'], epoch)
            writer.add_scalar('br/pct_near_levels', info_dict['avg_pct_near'], epoch)
            writer.add_scalar('br/quantization_mse', info_dict['avg_quantization_mse'], epoch)
        
        # Log quantization scales
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            log_quantization_scales(writer, model, epoch)
        
        # Log histograms periodically
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  Logging activation histograms...")
            # Run a forward pass to collect activations
            model.eval()
            hook_manager.set_training_mode(False)
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    if batch_idx >= 5:  # Only 5 batches for ResNet18
                        break
                    data = data.to(device)
                    _ = model(data)
            log_activation_histograms(writer, hook_manager, epoch)
            model.train()
        
        writer.flush()
        
        # Terminal output
        print(f"Epoch {epoch+1}/{args.epochs} [{stage_name}] (LR={current_lr:.6f}):")
        print(f"  Train - Loss: {train_loss:.4f}, Task: {train_task_loss:.4f}, Reg: {train_reg_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Task: {test_task_loss:.4f}, Reg: {test_reg_loss:.4f}, Acc: {test_acc:.2f}%")
        if use_br:
            print(f"  BR Effectiveness: {info_dict['avg_effectiveness']:.1f}% "
                  f"(MeanDist={info_dict['avg_mean_distance']:.4f}, "
                  f"@Levels={info_dict['avg_pct_near']:.1f}%, "
                  f"MSE={info_dict['avg_quantization_mse']:.6f})")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_accuracy': test_acc,
                'best_accuracy': best_acc,
                'clip_value': args.clip_value,
                'num_bits': args.num_bits,
                'lambda_br': args.lambda_br,
                'warmup_epochs': args.warmup_epochs,
                'freeze_alpha': args.freeze_alpha,
                'br_backprop_to_alpha': args.br_backprop_to_alpha,
                'manual_uniform_levels': args.manual_uniform_levels,
                'pretrained_imagenet': use_imagenet,
            }
            if use_br:
                checkpoint_data.update({
                    'br_effectiveness': info_dict['avg_effectiveness'],
                    'br_mean_distance': info_dict['avg_mean_distance'],
                    'br_pct_at_levels': info_dict['avg_pct_near'],
                    'br_quantization_mse': info_dict['avg_quantization_mse'],
                })
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  ✓ New best: {best_acc:.2f}%")
        
        # Also save a latest checkpoint every epoch (not just best)
        # This allows resuming from any epoch, not just the best one
        latest_checkpoint_path = checkpoint_path.replace('.pth', '_latest.pth')
        latest_checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'test_accuracy': test_acc,
            'best_accuracy': best_acc,
            'clip_value': args.clip_value,
            'num_bits': args.num_bits,
            'lambda_br': args.lambda_br,
            'warmup_epochs': args.warmup_epochs,
            'freeze_alpha': args.freeze_alpha,
            'br_backprop_to_alpha': args.br_backprop_to_alpha,
            'manual_uniform_levels': args.manual_uniform_levels,
            'pretrained_imagenet': use_imagenet,
        }
        if use_br:
            latest_checkpoint_data.update({
                'br_effectiveness': info_dict['avg_effectiveness'],
                'br_mean_distance': info_dict['avg_mean_distance'],
                'br_pct_at_levels': info_dict['avg_pct_near'],
                'br_quantization_mse': info_dict['avg_quantization_mse'],
            })
        torch.save(latest_checkpoint_data, latest_checkpoint_path)
        
        print()
    
    writer.close()
    
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    if use_br:
        print(f"\nFinal BR Metrics:")
        print(f"  Effectiveness: {info_dict['avg_effectiveness']:.1f}%")
        print(f"  Mean Distance: {info_dict['avg_mean_distance']:.6f}")
        print(f"  % @ Levels: {info_dict['avg_pct_near']:.1f}%")
        print(f"  Quantization MSE: {info_dict['avg_quantization_mse']:.6f}")
    
    print(f"\nModel saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

