#!/usr/bin/env python3
"""
MNIST ResNet18 with QAT + Bin Regularization (AWS/Slurm Optimized)

Optimized for batch jobs:
- Proper logging to file + stdout (for slurm-XXXX.out)
- No histograms or plotting (faster, less memory)
- Essential scalar metrics only

Usage:
    python experiments/mnist_resnet18_qat_binreg_aws.py \
        --pretrained checkpoints/mnist_resnet18_baseline_XXX.pth \
        --num-bits 2 --clip-value 6.0 --lambda-br 0.5 \
        --warmup-epochs 5 --epochs 30 --gpu 0 \
        --log-file logs/mnist_resnet18_qat_br.log
"""

import sys
import os
import logging
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

from abr.lsq_quantizer import QuantizedClippedReLU
from abr.regularizer_binreg import BinRegularizer
from abr.hooks import ActivationHookManager


def get_logger(log_file: str, name: str = "mnist_resnet18_qat_br") -> logging.Logger:
    """
    Setup logger that writes to both file and stdout (for slurm).
    
    Args:
        log_file: Path to log file
        name: Logger name
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if script is re-run
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Stream handler (stdout -> slurm-XXXX.out)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    return logger


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


def get_resnet18_mnist_qat(num_classes=10, pretrained_imagenet=True, clip_value=6.0, num_bits=2, pretrained_baseline=None, logger=None):
    """
    Get ResNet18 with QAT for MNIST.
    
    Args:
        num_classes: Number of output classes
        pretrained_imagenet: Whether to use ImageNet pretrained weights
        clip_value: Clip value for quantized ReLU
        num_bits: Number of bits for quantization
        pretrained_baseline: Path to baseline checkpoint (optional, takes priority over pretrained_imagenet)
        logger: Logger instance
    
    Returns:
        ResNet18 with QuantizedClippedReLU layers
    """
    log = logger.info if logger else print
    
    # Load ResNet18
    if pretrained_imagenet and pretrained_baseline is None:
        log("  Loading ImageNet pretrained weights...")
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
    
    # Adapt for MNIST (28x28 grayscale images)
    # Replace conv1: 3 channels (RGB) -> 1 channel (grayscale), 7x7 stride=2 -> 3x3 stride=1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool (too aggressive for 28x28)
    model.maxpool = nn.Identity()
    # Replace final FC for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Replace all ReLU with QuantizedClippedReLU
    replace_relu_with_qrelu(model, clip_value, num_bits)
    
    # Load baseline weights if provided
    if pretrained_baseline:
        log(f"  Loading pretrained baseline from: {pretrained_baseline}")
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
                    log(f"    Skipping {k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        log(f"  ✓ Loaded {len(filtered_dict)} parameter tensors from baseline")
    
    return model


def get_mnist_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """Get MNIST train and test loaders."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, hook_manager, regularizer, lambda_br, device, use_br=True, br_backprop_to_alpha=False, logger=None):
    """Train for one epoch with optional BR."""
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
    
    avg_loss = total_loss / len(loader)
    avg_task_loss = total_task_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_task_loss, avg_reg_loss, accuracy


def test_epoch(model, loader, criterion, hook_manager, regularizer, lambda_br, device, use_br=True, logger=None):
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
    parser = argparse.ArgumentParser(description='MNIST ResNet18 QAT + Bin Regularization (AWS/Slurm)')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained baseline model')
    parser.add_argument('--pretrained-imagenet', action='store_true', help='Use ImageNet pretrained weights (if no --pretrained baseline)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for MNIST data')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, help='Total epochs (default: 30)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs (default: 5)')
    parser.add_argument('--freeze-alpha', action='store_true', help='[EXPERIMENTAL] Freeze alpha after warmup. NOT recommended by BR paper.')
    parser.add_argument('--br-backprop-to-alpha', action='store_true',
                        help='[PAPER-FAITHFUL] Allow BR loss to backprop into alpha/s.')
    parser.add_argument('--br-sample-layers', type=int, default=-1, help='Number of layers to apply BR to (default: -1 for all)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--num-bits', type=int, default=2, help='Quantization bits (default: 2)')
    parser.add_argument('--clip-value', type=float, default=6.0, help='ReLU clip value (default: 6.0)')
    parser.add_argument('--lambda-br', type=float, default=0.5, help='BR lambda (default: 0.5)')
    parser.add_argument('--manual-uniform-levels', action='store_true', 
                        help='[NOT RECOMMENDED] Force uniform levels and freeze alpha.')
    parser.add_argument('--log-file', type=str, default='logs/mnist_resnet18_qat_br.log', help='Log file path')
    parser.add_argument('--log-interval', type=int, default=50, help='Batches between logging within an epoch')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Setup logger
    log_file = os.path.abspath(args.log_file)
    logger = get_logger(log_file)
    
    logger.info("="*70)
    logger.info("MNIST ResNet18 with QAT + Bin Regularization (AWS/Slurm)")
    logger.info("="*70)
    logger.info(f"Arguments: {vars(args)}")
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Quantization bits: {args.num_bits}")
    logger.info(f"Clip value: {args.clip_value}")
    logger.info(f"Lambda BR: {args.lambda_br}")
    logger.info(f"Manual uniform levels: {args.manual_uniform_levels}")
    logger.info(f"Freeze alpha after warmup: {args.freeze_alpha}")
    logger.info(f"BR backprop to alpha: {args.br_backprop_to_alpha}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info("="*70)
    
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
        logger.info("\n" + "="*70)
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info("="*70)
        logger.info(f"Loading checkpoint: {args.resume}")
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
        
        logger.info(f"  Resuming from epoch {start_epoch} (saved epoch: {saved_epoch})")
        logger.info(f"  Best accuracy so far: {best_acc:.2f}%")
        logger.info(f"  Using saved hyperparameters:")
        logger.info(f"    clip_value={args.clip_value}, num_bits={args.num_bits}")
        logger.info(f"    lambda_br={args.lambda_br}, warmup_epochs={args.warmup_epochs}")
        logger.info("="*70 + "\n")
    else:
        use_imagenet = args.pretrained_imagenet and args.pretrained is None
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    log_dir = f'./runs/mnist_resnet18_qat_binreg_{timestamp}'
    checkpoint_path = f'./checkpoints/mnist_resnet18_qat_binreg_{timestamp}.pth'
    
    # If resuming, use same checkpoint path (overwrite existing checkpoint)
    if args.resume:
        checkpoint_path = args.resume
        logger.info(f"  Will save to same checkpoint: {checkpoint_path}")
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Data
    logger.info(f"Loading MNIST dataset from: {args.data_dir}")
    train_loader, test_loader = get_mnist_loaders(args.batch_size, data_dir=args.data_dir)
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Model with QAT
    model = get_resnet18_mnist_qat(
        num_classes=10,
        pretrained_imagenet=use_imagenet,
        clip_value=args.clip_value,
        num_bits=args.num_bits,
        pretrained_baseline=args.pretrained if not args.resume else None,
        logger=logger
    ).to(device)
    
    logger.info("Model: ResNet18_QAT (adapted for MNIST + QuantizedClippedReLU)")
    if args.pretrained:
        logger.info(f"  ✓ Loaded baseline weights from: {args.pretrained}")
    elif use_imagenet:
        logger.info(f"  ✓ Started from ImageNet pretrained weights")
    else:
        logger.info(f"  ✓ Training from scratch")
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params:,}")
    
    # Count QuantizedClippedReLU layers
    qrelu_layers = [name for name, m in model.named_modules() if isinstance(m, QuantizedClippedReLU)]
    logger.info(f"Total QuantizedClippedReLU layers: {len(qrelu_layers)}")
    
    # Determine which layers to apply BR to
    if args.br_sample_layers > 0 and args.br_sample_layers < len(qrelu_layers):
        # Sample evenly spaced layers
        step = len(qrelu_layers) // args.br_sample_layers
        br_layer_names = [qrelu_layers[i*step] for i in range(args.br_sample_layers)]
        logger.info(f"✓ Applying BR to {len(br_layer_names)} sampled layers (every {step}th layer)")
    else:
        br_layer_names = qrelu_layers
        logger.info(f"✓ Applying BR to ALL {len(br_layer_names)} layers")
    
    # Manual uniform level initialization
    if args.manual_uniform_levels:
        logger.info("="*70)
        logger.info("MANUAL UNIFORM LEVEL INITIALIZATION")
        logger.info("="*70)
        Qp = 2**args.num_bits - 1
        alpha_uniform = args.clip_value / Qp
        logger.info(f"Setting alpha = {args.clip_value} / {Qp} = {alpha_uniform:.6f}")
        logger.info(f"Alpha will be FROZEN")
        logger.info("="*70)
        
        for name, module in model.named_modules():
            if isinstance(module, QuantizedClippedReLU):
                module.quantizer.alpha.data.fill_(alpha_uniform)
                module.quantizer.init_state.fill_(1)
                module.quantizer.alpha.requires_grad = False
        
        logger.info("✓ All alpha values set to uniform spacing and frozen")
        logger.info("="*70)
    
    # Hook manager for BR
    hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],
        layer_names=br_layer_names,
        exclude_first_last=False,
        detach_activations=False
    )
    
    logger.info(f"✓ Hooked {len(hook_manager.registered_layers)} layers for BR")
    
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
        logger.info("\n" + "="*70)
        logger.info("LOADING CHECKPOINT STATE")
        logger.info("="*70)
        
        # Load model state
        if 'model_state_dict' in resume_checkpoint:
            model.load_state_dict(resume_checkpoint['model_state_dict'])
            logger.info("  ✓ Loaded model state")
        else:
            logger.info("  ⚠ Warning: No model_state_dict in checkpoint, using current model")
        
        # Load optimizer state
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            logger.info("  ✓ Loaded optimizer state")
        else:
            logger.info("  ⚠ Warning: No optimizer_state_dict in checkpoint")
        
        # Load scheduler state (if saved)
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            logger.info("  ✓ Loaded scheduler state")
        else:
            # Manually step scheduler to correct position
            for _ in range(max(0, start_epoch - args.warmup_epochs)):
                scheduler.step()
            logger.info(f"  ✓ Scheduler stepped to epoch {start_epoch} position")
        
        # Restore alpha freeze state if applicable
        if resume_checkpoint.get('freeze_alpha', False):
            if start_epoch > resume_checkpoint.get('warmup_epochs', 0):
                logger.info("  ✓ Restoring alpha freeze state...")
                for name, module in model.named_modules():
                    if isinstance(module, QuantizedClippedReLU):
                        module.quantizer.alpha.requires_grad = False
        
        logger.info("="*70 + "\n")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs: {log_dir}")
    
    if args.resume:
        logger.info("="*70)
        logger.info(f"RESUMING TRAINING from epoch {start_epoch}/{args.epochs}")
        logger.info("="*70)
    else:
        logger.info("="*70)
        logger.info("Starting Two-Stage Training...")
        logger.info("="*70)
        logger.info("Stage 1: Warmup - Learn LSQ scales only (no BR)")
        logger.info("Stage 2: BR Training - Add bin regularization")
        logger.info("="*70)
    
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
            logger.info("="*70)
            logger.info("FREEZING ALPHA (preventing LSQ from escaping BR)")
            logger.info("="*70)
            for name, module in model.named_modules():
                if isinstance(module, QuantizedClippedReLU):
                    module.quantizer.alpha.requires_grad = False
                    alpha_val = module.quantizer.alpha.item()
                    logger.info(f"  {name}: alpha={alpha_val:.4f} (FROZEN)")
            logger.info("="*70)
        
        logger.info(f"==== Epoch {epoch+1}/{args.epochs} [{stage_name}] ====")
        
        # Train
        train_loss, train_task_loss, train_reg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, hook_manager, regularizer, args.lambda_br, device, 
            use_br=use_br, br_backprop_to_alpha=args.br_backprop_to_alpha, logger=logger
        )
        
        # Test
        test_loss, test_task_loss, test_reg_loss, test_acc, info_dict = test_epoch(
            model, test_loader, criterion, hook_manager, regularizer, args.lambda_br, device, use_br=use_br, logger=logger
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
        
        writer.flush()
        
        # Terminal/log output
        logger.info(f"Epoch {epoch+1}/{args.epochs} [{stage_name}] (LR={current_lr:.6f}):")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Task: {train_task_loss:.4f}, Reg: {train_reg_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Test  - Loss: {test_loss:.4f}, Task: {test_task_loss:.4f}, Reg: {test_reg_loss:.4f}, Acc: {test_acc:.2f}%")
        if use_br:
            logger.info(f"  BR Effectiveness: {info_dict['avg_effectiveness']:.1f}% "
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
            logger.info(f"  ✓ New best: {best_acc:.2f}% - Model saved")
    
    writer.close()
    
    logger.info("="*70)
    logger.info("Training Complete!")
    logger.info("="*70)
    logger.info(f"Best Test Accuracy: {best_acc:.2f}%")
    if use_br:
        logger.info(f"\nFinal BR Metrics:")
        logger.info(f"  Effectiveness: {info_dict['avg_effectiveness']:.1f}%")
        logger.info(f"  Mean Distance: {info_dict['avg_mean_distance']:.6f}")
        logger.info(f"  % @ Levels: {info_dict['avg_pct_near']:.1f}%")
        logger.info(f"  Quantization MSE: {info_dict['avg_quantization_mse']:.6f}")
    
    logger.info(f"\nModel saved to: {checkpoint_path}")
    logger.info(f"Logs written to: {log_file}")
    logger.info(f"TensorBoard logs: {log_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

