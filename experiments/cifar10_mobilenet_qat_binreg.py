#!/usr/bin/env python3
"""
CIFAR-10 MobileNetV2 with QAT + Bin Regularization

Fine-tune MobileNetV2 on CIFAR-10 with:
- LSQ (Learned Step-size Quantization) for activations
- Bin Regularization to cluster activations at quantization levels
- 2-bit quantization

This tests if BR scales to larger models with BatchNorm.

Usage:
    python experiments/cifar10_mobilenet_qat_binreg.py \
        --pretrained checkpoints/cifar10_mobilenet_baseline_XXX.pth \
        --num-bits 2 --clip-value 6.0 --lambda-br 2.0 \
        --manual-uniform-levels --epochs 30 --gpu 0
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
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm

from abr.lsq_quantizer import QuantizedClippedReLU
from abr.regularizer_binreg import BinRegularizer
from abr.hooks import ActivationHookManager


def replace_relu6_with_qrelu(module, clip_value, num_bits):
    """
    Recursively replace all ReLU6 layers with QuantizedClippedReLU.
    
    Args:
        module: PyTorch module to modify
        clip_value: Clip value for QuantizedClippedReLU
        num_bits: Number of bits for quantization
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU6):
            # Replace ReLU6 with QuantizedClippedReLU
            setattr(module, name, QuantizedClippedReLU(clip_value, num_bits))
        else:
            # Recursively apply to children
            replace_relu6_with_qrelu(child, clip_value, num_bits)


def get_mobilenetv2_cifar10_qat(num_classes=10, clip_value=6.0, num_bits=2, pretrained_baseline=None):
    """
    Get MobileNetV2 with QAT for CIFAR-10.
    
    Args:
        num_classes: Number of output classes
        clip_value: Clip value for quantized ReLU
        num_bits: Number of bits for quantization
        pretrained_baseline: Path to baseline checkpoint (optional)
    
    Returns:
        MobileNetV2 with QuantizedClippedReLU layers
    """
    # Load pre-defined MobileNetV2 (no pretrained weights, will load baseline later)
    model = mobilenet_v2(weights=None)
    
    # Modify for CIFAR-10
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # Replace all ReLU6 with QuantizedClippedReLU
    replace_relu6_with_qrelu(model, clip_value, num_bits)
    
    # Load baseline weights if provided
    if pretrained_baseline:
        print(f"Loading pretrained baseline from: {pretrained_baseline}")
        checkpoint = torch.load(pretrained_baseline, map_location='cpu')
        
        # Filter out keys that don't match (quantizer parameters)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if 'quantizer' not in k and k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"  Skipping {k} (shape mismatch)")
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"✓ Loaded {len(filtered_dict)} parameter tensors")
    
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


def train_epoch(model, loader, optimizer, criterion, hook_manager, regularizer, lambda_br, device, use_br=True):
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
            
            # Extract alphas from model (using FULL layer names to match activations dict)
            alphas = {}
            for name, module in model.named_modules():
                if isinstance(module, QuantizedClippedReLU):
                    # Use FULL layer name (not just last component)
                    alphas[name] = module.quantizer.alpha.item()
            
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
                
                # Extract alphas (using FULL layer names to match activations dict)
                alphas = {}
                for name, module in model.named_modules():
                    if isinstance(module, QuantizedClippedReLU):
                        # Use FULL layer name (not just last component)
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
    parser = argparse.ArgumentParser(description='CIFAR-10 MobileNetV2 QAT + Bin Regularization')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained baseline model')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, help='Total epochs (default: 30)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--num-bits', type=int, default=2, help='Quantization bits (default: 2)')
    parser.add_argument('--clip-value', type=float, default=6.0, help='ReLU clip value (default: 6.0)')
    parser.add_argument('--lambda-br', type=float, default=2.0, help='BR lambda (default: 2.0)')
    parser.add_argument('--manual-uniform-levels', action='store_true', help='Use manual uniform quantization levels')
    parser.add_argument('--br-sample-layers', type=int, default=10, help='Number of layers to apply BR to (default: 10, set to -1 for all)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("CIFAR-10 MobileNetV2 with QAT + Bin Regularization")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Quantization bits: {args.num_bits}")
    print(f"Clip value: {args.clip_value}")
    print(f"Lambda BR: {args.lambda_br}")
    print(f"Manual uniform levels: {args.manual_uniform_levels}")
    print("="*70)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    log_dir = f'./runs/cifar10_mobilenet_qat_binreg_{timestamp}'
    checkpoint_path = f'./checkpoints/cifar10_mobilenet_qat_binreg_{timestamp}.pth'
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    
    # Model with QAT (using torchvision's MobileNetV2 + quantized ReLU)
    model = get_mobilenetv2_cifar10_qat(
        num_classes=10,
        clip_value=args.clip_value,
        num_bits=args.num_bits,
        pretrained_baseline=args.pretrained
    ).to(device)
    
    print(f"\nModel: MobileNetV2_QAT (from torchvision + QuantizedClippedReLU)")
    if args.pretrained:
        print(f"  ✓ Loaded baseline weights from: {args.pretrained}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Count QuantizedClippedReLU layers (ACTIVATION layers only)
    qrelu_layers = [name for name, m in model.named_modules() if isinstance(m, QuantizedClippedReLU)]
    print(f"Total QuantizedClippedReLU (activation) layers: {len(qrelu_layers)}")
    
    # Show first few activation layer names for verification
    print(f"\nExample activation layers (first 5):")
    for i, name in enumerate(qrelu_layers[:5]):
        print(f"  [{i+1}] {name}")
    if len(qrelu_layers) > 5:
        print(f"  ... and {len(qrelu_layers) - 5} more")
    print()
    
    # Determine which layers to apply BR to
    if args.br_sample_layers > 0 and args.br_sample_layers < len(qrelu_layers):
        # Sample evenly spaced layers
        step = len(qrelu_layers) // args.br_sample_layers
        br_layer_names = [qrelu_layers[i*step] for i in range(args.br_sample_layers)]
        print(f"✓ Applying BR to {len(br_layer_names)} sampled ACTIVATION layers (every {step}th layer)")
    else:
        br_layer_names = qrelu_layers
        print(f"✓ Applying BR to ALL {len(br_layer_names)} ACTIVATION layers")
    
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
    
    # Hook manager for BR (targets ACTIVATION layers only, not Conv/Linear)
    hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],  # Only QuantizedClippedReLU (activations)
        layer_names=br_layer_names,
        exclude_first_last=False,
        detach_activations=False
    )
    
    import sys
    print(f"\n✓ Hooked {len(hook_manager.registered_layers)} ACTIVATION layers for BR:")
    for i, layer in enumerate(hook_manager.registered_layers[:5], 1):
        print(f"  [{i}] {layer}")
    if len(hook_manager.registered_layers) > 5:
        print(f"  ... and {len(hook_manager.registered_layers) - 5} more")
    print(f"\nNote: BR is applied ONLY to QuantizedClippedReLU (activation) layers,")
    print(f"      NOT to Conv2d, Linear, or BatchNorm layers.")
    sys.stdout.flush()  # Force print to show
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
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    print("="*70)
    print("Starting Training...")
    print("="*70)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Determine stage
        if args.manual_uniform_levels:
            is_warmup = False
            use_br = True
            stage_name = "BR TRAINING"
        else:
            is_warmup = (epoch < args.warmup_epochs)
            use_br = not is_warmup
            stage_name = "WARMUP" if is_warmup else "BR TRAINING"
        
        # Train
        train_loss, train_task_loss, train_reg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, hook_manager, regularizer, args.lambda_br, device, use_br=use_br
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
                    if batch_idx >= 10:  # Only 10 batches
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
                'test_accuracy': test_acc,
                'best_accuracy': best_acc,
                'clip_value': args.clip_value,
                'num_bits': args.num_bits,
                'lambda_br': args.lambda_br,
                'manual_uniform_levels': args.manual_uniform_levels,
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
        
        # Comparison with MNIST
        print(f"\n  Comparison:")
        print(f"    MNIST (no BN):  91.0% effectiveness")
        print(f"    MNIST (with BN): 85.5% effectiveness")
        print(f"    CIFAR-10 MobileNetV2: {info_dict['avg_effectiveness']:.1f}% effectiveness")
        
        if info_dict['avg_effectiveness'] > 80:
            print(f"    ✅ BR scales to larger models with BN!")
        elif info_dict['avg_effectiveness'] > 70:
            print(f"    ⚠️  Reduced but still decent (larger model, more BN layers)")
        else:
            print(f"    ❌ Effectiveness dropped significantly")
    
    print(f"\nModel saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

