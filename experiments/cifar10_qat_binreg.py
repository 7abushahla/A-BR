"""
CIFAR-10 QAT with LSQ + Bin Regularization (BR)

Same BR approach as MNIST but on CIFAR-10 with a simple CNN.

Usage:
    python experiments/cifar10_qat_binreg.py --num-bits 2 --warmup-epochs 30 --epochs 100 --lambda-br 10.0
"""

import os
import sys
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abr.hooks import ActivationHookManager
from abr.regularizer_binreg import BinRegularizer
from abr.lsq_quantizer import QuantizedClippedReLU

# Use matplotlib backend that doesn't require display
import matplotlib
matplotlib.use('Agg')


# ======================== Model Definition ========================

class SimpleCNN_CIFAR10_QAT(nn.Module):
    """
    Simple CNN for CIFAR-10 with QAT (Quantized Clipped ReLU activations).
    Same architecture as baseline but with learnable quantization.
    """
    
    def __init__(self, num_classes=10, base=32, clip_value=1.0, num_bits=2):
        super().__init__()
        
        self.clip_value = clip_value
        self.num_bits = num_bits
        
        # Input: 32x32x3
        self.conv1 = nn.Conv2d(3, base, 3, stride=2, padding=1, bias=True)
        self.relu1 = QuantizedClippedReLU(clip_value, num_bits)
        
        self.conv2 = nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=True)
        self.relu2 = QuantizedClippedReLU(clip_value, num_bits)
        
        self.conv3 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=True)
        self.relu3 = QuantizedClippedReLU(clip_value, num_bits)
        
        self.conv4 = nn.Conv2d(base*4, base*4, 3, stride=1, padding=1, bias=True)
        self.relu4 = QuantizedClippedReLU(clip_value, num_bits)
        
        # Head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(base*4*4*4, 128, bias=True)
        self.relu5 = QuantizedClippedReLU(clip_value, num_bits)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv blocks with quantized clipped ReLU
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        # Head
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        
        return x


# ======================== Data Loading ========================

def get_cifar10_loaders(batch_size=256, num_workers=4):
    """Get CIFAR-10 train and test loaders with standard augmentation."""
    
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


# ======================== Logging Functions (reuse from MNIST) ========================

def log_activation_histograms(writer, hook_manager, epoch, model=None):
    """Log activation histograms to TensorBoard (same as MNIST)."""
    post_quant_activations = hook_manager.get_activations()
    pre_quant_activations = hook_manager.get_pre_quant_activations()
    
    quant_modules = {}
    if model is not None:
        for name, module in model.named_modules():
            if isinstance(module, QuantizedClippedReLU):
                quant_modules[name] = module
    
    for name in post_quant_activations.keys():
        post_acts_flat = post_quant_activations[name].flatten()
        writer.add_histogram(f'activations_post_quant/{name}', post_acts_flat, epoch)
        
        if name in pre_quant_activations:
            pre_acts_flat = pre_quant_activations[name].flatten()
            writer.add_histogram(f'activations_pre_quant/{name}', pre_acts_flat, epoch)
            
            if name in quant_modules:
                levels = quant_modules[name].quantizer.get_quantization_levels()
                max_level = levels[-1].item()
                
                pre_acts_zoomed = torch.clamp(pre_acts_flat, 0, max_level)
                writer.add_histogram(f'activations_pre_quant_ZOOMED/{name}', pre_acts_zoomed, epoch)
                writer.add_histogram(f'activations_post_quant_ZOOMED/{name}', post_acts_flat, epoch)
            
            residual = torch.abs(pre_acts_flat - post_acts_flat)
            writer.add_histogram(f'quant_residual/{name}', residual, epoch)
            
            writer.add_scalar(f'quant_residual_stats/{name}/mean', residual.mean().item(), epoch)
            writer.add_scalar(f'quant_residual_stats/{name}/max', residual.max().item(), epoch)
            writer.add_scalar(f'quant_residual_stats/{name}/std', residual.std().item(), epoch)


def log_quantization_scales(writer, model, epoch):
    """Log learned quantization scales (alpha)."""
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            alpha = module.quantizer.alpha.item()
            writer.add_scalar(f'quant_scales/{name}', alpha, epoch)
            
            levels = module.quantizer.get_quantization_levels()
            for i, level in enumerate(levels):
                writer.add_scalar(f'quant_levels/{name}/level_{i}', level.item(), epoch)


def log_binreg_scalars(writer, info_dict, epoch):
    """Log bin regularization metrics to TensorBoard."""
    writer.add_scalar('binreg/avg_loss', info_dict['avg_loss'], epoch)
    writer.add_scalar('binreg/br_mse_loss', info_dict['avg_mse'], epoch)
    writer.add_scalar('binreg/br_var_loss', info_dict['avg_var'], epoch)
    writer.add_scalar('binreg/quantization_mse', info_dict.get('avg_quantization_mse', 0), epoch)
    
    writer.add_scalar('binreg/effectiveness', info_dict['avg_effectiveness'], epoch)
    writer.add_scalar('binreg/mean_distance', info_dict['avg_mean_distance'], epoch)
    writer.add_scalar('binreg/pct_near_levels', info_dict['avg_pct_near'], epoch)
    
    for layer_name, layer_info in info_dict['layer_losses'].items():
        writer.add_scalar(f'binreg_layer/{layer_name}/loss', layer_info['loss'], epoch)
        writer.add_scalar(f'binreg_layer/{layer_name}/mse', layer_info['mse'], epoch)
        writer.add_scalar(f'binreg_layer/{layer_name}/var', layer_info['var'], epoch)
        writer.add_scalar(f'binreg_layer/{layer_name}/effectiveness', layer_info['effectiveness'], epoch)


# ======================== Helper Functions ========================

def get_layer_alphas(model, layer_names, detach=True):
    """Extract current alpha values from LSQ quantizers."""
    alphas = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU) and name in layer_names:
            if detach:
                alphas[name] = module.quantizer.alpha.item()
            else:
                alphas[name] = module.quantizer.alpha.squeeze()
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
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        task_loss = criterion(output, target)
        
        if use_br:
            pre_quant_activations = hook_manager.get_pre_quant_activations()
            alphas = get_layer_alphas(model, hook_manager.registered_layers, detach=(not br_backprop_to_alpha))
            br_loss, _ = regularizer.compute_total_loss(pre_quant_activations, alphas)
            total_loss_value = task_loss + lambda_br * br_loss
            reg_loss_value = (lambda_br * br_loss).item() if isinstance(br_loss, torch.Tensor) else (lambda_br * br_loss)
        else:
            total_loss_value = task_loss
            reg_loss_value = 0.0
        
        total_loss_value.backward()
        optimizer.step()
        
        total_loss += total_loss_value.item()
        total_task_loss += task_loss.item()
        total_reg_loss += reg_loss_value
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({'loss': total_loss_value.item(), 'acc': 100. * correct / total})
    
    return (total_loss / len(loader), 
            total_task_loss / len(loader),
            total_reg_loss / len(loader),
            100. * correct / total)


def test_epoch(model, loader, criterion, hook_manager, regularizer, lambda_br, device, use_br=True):
    """Evaluate on test set."""
    model.eval()
    hook_manager.set_training_mode(False)
    
    test_loss = 0
    test_task_loss = 0
    test_reg_loss = 0
    correct = 0
    total = 0
    
    accumulated_effectiveness = 0
    accumulated_mean_distance = 0
    accumulated_pct_near = 0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            task_loss = criterion(output, target)
            
            if use_br:
                activations = hook_manager.get_pre_quant_activations()
                alphas = get_layer_alphas(model, hook_manager.registered_layers, detach=True)
                br_loss, info_dict = regularizer.compute_total_loss(activations, alphas)
                loss = task_loss + lambda_br * br_loss
                reg_loss_value = (lambda_br * br_loss).item() if isinstance(br_loss, torch.Tensor) else (lambda_br * br_loss)
                
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
    
    if use_br and num_batches > 0:
        info_dict = {
            'avg_loss': test_reg_loss / len(loader),
            'avg_effectiveness': accumulated_effectiveness / num_batches,
            'avg_mean_distance': accumulated_mean_distance / num_batches,
            'avg_pct_near': accumulated_pct_near / num_batches,
            'avg_quantization_mse': (accumulated_mean_distance / num_batches) ** 2,
            'avg_mse': 0,
            'avg_var': 0,
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
    parser = argparse.ArgumentParser(description='CIFAR-10 QAT with LSQ + Bin Regularization')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs (default: 100)')
    parser.add_argument('--warmup-epochs', type=int, default=30, help='Warmup epochs (default: 30)')
    parser.add_argument('--freeze-alpha', action='store_true', help='Freeze alpha after warmup')
    parser.add_argument('--br-backprop-to-alpha', action='store_true', help='Allow BR to backprop into alpha')
    parser.add_argument('--br-all-layers', action='store_true', help='Apply BR to ALL layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--num-bits', type=int, default=2, help='Target bit-width (default: 2)')
    parser.add_argument('--clip-value', type=float, default=1.0, help='ReLU clip value (default: 1.0)')
    parser.add_argument('--lambda-br', type=float, default=10.0, help='Lambda for BR loss (default: 10.0)')
    parser.add_argument('--pretrained-baseline', type=str, default=None, help='Path to baseline FP32 checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-save', action='store_true', help='Do not save checkpoints (for testing)')
    args = parser.parse_args()
    
    # Set random seeds
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
    print("CIFAR-10 QAT with LSQ + Bin Regularization")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Clip value: {args.clip_value}")
    print(f"Bit-width: {args.num_bits} bits ({2**args.num_bits} levels)")
    print(f"Lambda BR: {args.lambda_br}")
    print(f"Learning rate: {args.lr}")
    print(f"Total epochs: {args.epochs}")
    print(f"  - Warmup: {args.warmup_epochs} epochs")
    print(f"  - Fine-tune: {args.epochs - args.warmup_epochs} epochs")
    if args.freeze_alpha:
        print(f"  ⚠️  Alpha will be frozen after warmup")
    if args.br_backprop_to_alpha:
        print(f"  🔬 BR backprop to alpha: ENABLED")
    print("="*70)
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    log_dir = f'./runs/cifar10_qat_binreg_{timestamp}'
    checkpoint_path = f'./checkpoints/cifar10_qat_binreg_{timestamp}.pth'
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(args.batch_size, num_workers=4)
    
    # Model
    model = SimpleCNN_CIFAR10_QAT(
        num_classes=10,
        base=32,
        clip_value=args.clip_value,
        num_bits=args.num_bits
    ).to(device)
    
    # Optional: warm-start from baseline
    if args.pretrained_baseline is not None:
        print(f"\nLoading baseline weights from: {args.pretrained_baseline}")
        base_ckpt = torch.load(args.pretrained_baseline, map_location=device)
        base_sd = base_ckpt['model_state_dict'] if 'model_state_dict' in base_ckpt else base_ckpt
        missing, unexpected = model.load_state_dict(base_sd, strict=False)
        print(f"✓ Loaded baseline weights (strict=False)")
        if missing:
            print(f"  Missing keys (expected QAT params): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    
    print(f"\nModel: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    print(f"\nInitial quantization scales (LSQ):")
    for name, module in model.named_modules():
        if isinstance(module, QuantizedClippedReLU):
            print(f"  {name}: alpha={module.quantizer.alpha.item():.4f}")
    
    # Hook managers
    if args.br_all_layers:
        br_layers = ['relu1', 'relu2', 'relu3', 'relu4', 'relu5']
        print("\nBR will be applied to ALL layers")
    else:
        br_layers = ['relu2', 'relu3', 'relu4']
        print("\nBR will be applied to middle layers only (relu2-relu4)")
    
    hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],
        layer_names=br_layers,
        exclude_first_last=False,
        detach_activations=False
    )
    
    viz_hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],
        layer_names=['relu1', 'relu2', 'relu3', 'relu4', 'relu5'],
        exclude_first_last=False,
        detach_activations=True
    )
    
    # Bin regularizer
    regularizer = BinRegularizer(num_bits=args.num_bits)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Use ReduceLROnPlateau - reduces LR when validation accuracy plateaus
    # More adaptive than CosineAnnealing, better for fine-tuning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # Maximize test accuracy
        factor=0.5,           # Reduce LR by half
        patience=10,          # Wait 10 epochs before reducing
        min_lr=1e-6           # Don't go below this
    )
    print(f"  Learning rate scheduler: ReduceLROnPlateau (patience={10}, factor={0.5})")
    
    print("\n" + "="*70)
    print("Starting Two-Stage Training")
    print("="*70)
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        is_warmup = (epoch < args.warmup_epochs)
        use_br = not is_warmup
        stage_name = "WARMUP (scales only)" if is_warmup else "FINE-TUNE (scales + BR)"
        
        # Freeze alpha after warmup if requested
        if args.freeze_alpha and epoch == args.warmup_epochs:
            print("\n" + "="*70)
            print("FREEZING ALPHA")
            print("="*70)
            for name, module in model.named_modules():
                if isinstance(module, QuantizedClippedReLU):
                    module.quantizer.alpha.requires_grad = False
                    print(f"  {name}: alpha={module.quantizer.alpha.item():.4f} (FROZEN)")
            print("="*70 + "\n")
        
        # Train
        train_loss, train_task_loss, train_reg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, hook_manager, regularizer, 
            args.lambda_br, device, use_br=use_br, br_backprop_to_alpha=args.br_backprop_to_alpha
        )
        
        # Test
        test_loss, test_task_loss, test_reg_loss, test_acc, info_dict = test_epoch(
            model, test_loader, criterion, hook_manager, regularizer, 
            args.lambda_br, device, use_br=use_br
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
        
        log_quantization_scales(writer, model, epoch)
        
        if use_br:
            log_binreg_scalars(writer, info_dict, epoch)
        
        # Log histograms every 5 epochs
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            log_activation_histograms(writer, viz_hook_manager, epoch, model=model)
        
        writer.flush()
        
        # Terminal output
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} [{stage_name}] (LR={current_lr:.6f}):")
        print(f"  Train - Loss: {train_loss:.4f}, Task: {train_task_loss:.4f}, Reg: {train_reg_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Task: {test_task_loss:.4f}, Reg: {test_reg_loss:.4f}, Acc: {test_acc:.2f}%")
        if use_br:
            print(f"  BR Effectiveness: {info_dict['avg_effectiveness']:.1f}% "
                  f"(MeanDist={info_dict['avg_mean_distance']:.4f}, "
                  f"@Levels={info_dict['avg_pct_near']:.1f}%)")
        
        # Step scheduler (ReduceLROnPlateau needs the metric)
        scheduler.step(test_acc)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', current_lr, epoch)
        
        # Save best model
        if test_acc > best_acc and not args.no_save:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'best_acc': best_acc,
                'clip_value': args.clip_value,
                'num_bits': args.num_bits,
                'lambda_br': args.lambda_br,
                'warmup_epochs': args.warmup_epochs,
                'freeze_alpha': args.freeze_alpha,
            }, checkpoint_path)
            print(f"  ✓ New best: {best_acc:.2f}%")
        
        print()
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    if not args.no_save:
        print(f"Model saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")
    if use_br:
        print(f"\nFinal BR Effectiveness: {info_dict['avg_effectiveness']:.1f}%")
        print(f"  - Mean Distance: {info_dict['avg_mean_distance']:.6f}")
        print(f"  - Quantization MSE: {info_dict.get('avg_quantization_mse', 0):.8f}")
    print("="*70)
    
    writer.close()


if __name__ == '__main__':
    main()

