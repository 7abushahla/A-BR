#!/usr/bin/env python3
"""
Automated QAT+BR Training + Evaluation Pipeline for CIFAR-10 ResNet18

This script automates the full QAT+BR workflow:
1. Load baseline checkpoints from PTQ sweep
2. Fine-tune each with QAT+BR for multiple (bit_width, lambda) combinations
3. Evaluate each QAT model (FP32 accuracy, INT accuracy, MSE)
4. Aggregate results (mean ± std across seeds)
5. Generate formatted tables

Usage:
    # ReLU (clip-value None) with 3 seeds
    python experiments/cifar10_automated_qat_br_sweep.py \
        --baseline-checkpoints-dir results/relu_sweep/checkpoints \
        --seeds 42 43 44 \
        --bit-widths 1 2 4 \
        --lambdas 0.1 1.0 10.0 \
        --qat-epochs 30 \
        --warmup-epochs 5 \
        --gpu 0 \
        --output-dir results/relu_qat_br_sweep

    # ReLU6 (clip-value 6.0) with 3 seeds
    python experiments/cifar10_automated_qat_br_sweep.py \
        --baseline-checkpoints-dir results/relu6_sweep/checkpoints \
        --seeds 42 43 44 \
        --bit-widths 1 2 4 \
        --lambdas 0.1 1.0 10.0 \
        --qat-epochs 30 \
        --warmup-epochs 5 \
        --gpu 0 \
        --output-dir results/relu6_qat_br_sweep
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd
from glob import glob

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Import necessary components
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import QAT+BR components
from abr.lsq_quantizer import QuantizedClippedReLU
from abr.regularizer_binreg import BinRegularizer
from abr.hooks import ActivationHookManager
from experiments.cifar10_resnet18_qat_binreg import (
    get_resnet18_cifar10_qat,
    get_cifar10_loaders
)


class QATBRSweepRunner:
    """
    Manages the full QAT+BR training + evaluation pipeline.
    """
    
    def __init__(
        self,
        baseline_checkpoints_dir: str,
        seeds: List[int],
        bit_widths: List[int],
        lambdas: List[float],
        output_dir: str,
        device: str = 'cuda:0'
    ):
        self.baseline_checkpoints_dir = Path(baseline_checkpoints_dir)
        self.seeds = seeds
        self.bit_widths = bit_widths
        self.lambdas = lambdas
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
    def find_baseline_checkpoint(self, seed: int) -> Optional[str]:
        """Find baseline checkpoint for a given seed."""
        pattern = str(self.baseline_checkpoints_dir / f"*seed{seed}*.pth")
        matches = glob(pattern)
        if matches:
            return matches[0]  # Return first match
        return None
    
    def train_qat_br(
        self,
        baseline_checkpoint: str,
        num_bits: int,
        lambda_br: float,
        seed: int,
        args: argparse.Namespace
    ) -> Tuple[str, float]:
        """
        Fine-tune a baseline model with QAT+BR.
        
        Returns:
            checkpoint_path: Path to saved QAT model
            best_accuracy: Best test accuracy achieved
        """
        print(f"\n{'='*80}")
        print(f"QAT+BR TRAINING - Seed {seed}, {num_bits}-bit, λ={lambda_br}")
        print(f"{'='*80}")
        print(f"Baseline checkpoint: {baseline_checkpoint}")
        
        # Set random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Load baseline to get clip_value
        baseline = torch.load(baseline_checkpoint, map_location='cpu')
        clip_value = baseline.get('clip_value', None)
        
        # Create timestamp for unique checkpoint name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"qat_br_clip{clip_value}_seed{seed}_b{num_bits}_lam{lambda_br}_{timestamp}.pth"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        # Get data loaders
        train_loader, test_loader = get_cifar10_loaders(args.batch_size)
        
        # Create QAT model (loads baseline weights automatically)
        model = get_resnet18_cifar10_qat(
            num_classes=10,
            pretrained_imagenet=False,
            clip_value=clip_value if clip_value is not None else 6.0,
            num_bits=num_bits,
            pretrained_baseline=baseline_checkpoint
        ).to(self.device)
        
        print(f"Model: ResNet18 QAT (clip_value={clip_value}, {num_bits}-bit)")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup BR
        hook_manager = ActivationHookManager(
            model=model,
            target_modules=[QuantizedClippedReLU],
            layer_names=None,  # All layers
            exclude_first_last=False,
            detach_activations=False
        )
        
        regularizer = BinRegularizer(num_bits=num_bits)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        finetune_epochs = args.qat_epochs - args.warmup_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs, eta_min=args.lr * 0.01
        )
        
        # Training loop
        best_acc = 0.0
        for epoch in range(args.qat_epochs):
            is_warmup = (epoch < args.warmup_epochs)
            use_br = not is_warmup
            
            # Freeze alpha after warmup if requested
            if args.freeze_alpha and epoch == args.warmup_epochs:
                print(f"\n{'='*80}")
                print("FREEZING ALPHA")
                print(f"{'='*80}")
                for name, module in model.named_modules():
                    if isinstance(module, QuantizedClippedReLU):
                        module.quantizer.alpha.requires_grad = False
                print("✓ Alpha values frozen\n")
            
            # Train
            model.train()
            hook_manager.set_training_mode(True)
            hook_manager.clear_activations()
            train_loss = 0
            train_ce_loss = 0
            train_br_loss = 0
            train_correct = 0
            train_total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                ce_loss = criterion(output, target)
                
                # Bin regularization loss
                if use_br:
                    activations = hook_manager.get_pre_quant_activations()
                    alphas = {}
                    for name, module in model.named_modules():
                        if isinstance(module, QuantizedClippedReLU):
                            if args.br_backprop_to_alpha:
                                alphas[name] = module.quantizer.alpha.squeeze()  # Tensor, keeps gradient
                            else:
                                alphas[name] = module.quantizer.alpha.item()  # Python float, no gradient
                    
                    br_loss, info_dict = regularizer.compute_total_loss(activations, alphas)
                    loss = ce_loss + lambda_br * br_loss
                    reg_loss_val = br_loss if isinstance(br_loss, float) else br_loss.item()
                    train_br_loss += reg_loss_val
                else:
                    loss = ce_loss
                
                loss.backward()
                optimizer.step()
                hook_manager.clear_activations()
                
                train_loss += loss.item()
                train_ce_loss += ce_loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # Test
            model.eval()
            hook_manager.set_training_mode(False)
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    test_loss += loss.item()
                    _, predicted = output.max(1)
                    test_total += target.size(0)
                    test_correct += predicted.eq(target).sum().item()
            
            test_acc = 100. * test_correct / test_total
            
            # Step scheduler (after warmup)
            if epoch >= args.warmup_epochs:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            stage = "WARMUP" if is_warmup else "BR"
            
            if use_br:
                avg_br = train_br_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{args.qat_epochs} [{stage}] (LR={current_lr:.6f}): "
                      f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, BR Loss={avg_br:.6f}")
            else:
                print(f"Epoch {epoch+1}/{args.qat_epochs} [{stage}] (LR={current_lr:.6f}): "
                      f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy': test_acc,
                    'best_accuracy': best_acc,
                    'clip_value': clip_value,
                    'num_bits': num_bits,
                    'lambda_br': lambda_br,
                    'seed': seed,
                    'baseline_checkpoint': baseline_checkpoint,
                }, checkpoint_path)
        
        print(f"\n✓ QAT+BR training complete! Best Test Accuracy: {best_acc:.2f}%")
        print(f"✓ Model saved to: {checkpoint_path}")
        
        return str(checkpoint_path), best_acc
    
    def evaluate_qat_model(
        self,
        checkpoint_path: str,
        seed: int
    ) -> Tuple[float, float, float]:
        """
        Evaluate a QAT model.
        
        Returns:
            fp32_accuracy: FP32 test accuracy
            int_accuracy: Quantized accuracy
            avg_mse: Average MSE across layers
        """
        print(f"\n  Evaluating QAT model...")
        
        # Set seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        clip_value = checkpoint.get('clip_value', None)
        num_bits = checkpoint.get('num_bits', 2)
        
        model = get_resnet18_cifar10_qat(
            num_classes=10,
            pretrained_imagenet=False,
            clip_value=clip_value if clip_value is not None else 6.0,
            num_bits=num_bits,
            pretrained_baseline=None
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Data loader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
        
        # Hook manager to collect activations
        hook_manager = ActivationHookManager(
            model=model,
            target_modules=[QuantizedClippedReLU],
            layer_names=None,
            exclude_first_last=False,
            detach_activations=True
        )
        hook_manager.set_training_mode(False)
        
        # Evaluate
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        int_accuracy = 100. * correct / total
        
        # Collect activations for MSE calculation
        hook_manager.clear_activations()
        orig_acts_all = []
        quant_acts_all = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if batch_idx >= 3:  # Only first 3 batches
                    break
                data = data.to(self.device)
                _ = model(data)
                
                # Get activations
                quant_acts = hook_manager.get_activations()
                orig_acts = hook_manager.get_pre_quant_activations()
                
                # Flatten and store
                for name in orig_acts.keys():
                    orig_acts_all.append(orig_acts[name].flatten().cpu())
                    quant_acts_all.append(quant_acts[name].flatten().cpu())
                
                hook_manager.clear_activations()
        
        # Compute MSE
        if orig_acts_all and quant_acts_all:
            orig_tensor = torch.cat(orig_acts_all)
            quant_tensor = torch.cat(quant_acts_all)
            mse = ((orig_tensor - quant_tensor) ** 2).mean().item()
        else:
            mse = 0.0
        
        # FP32 accuracy (from baseline checkpoint)
        baseline_checkpoint = checkpoint.get('baseline_checkpoint', '')
        if baseline_checkpoint and Path(baseline_checkpoint).exists():
            baseline = torch.load(baseline_checkpoint, map_location='cpu')
            fp32_accuracy = baseline.get('best_accuracy', baseline.get('test_accuracy', 0.0))
        else:
            fp32_accuracy = 0.0
        
        print(f"    ✓ FP32 Acc: {fp32_accuracy:.2f}%, INT Acc: {int_accuracy:.2f}%, MSE: {mse:.6f}")
        
        return fp32_accuracy, int_accuracy, mse
    
    def run_full_sweep(self, args: argparse.Namespace):
        """
        Run the full QAT+BR sweep for all seeds.
        """
        print(f"\n{'='*80}")
        print(f"AUTOMATED QAT+BR SWEEP")
        print(f"{'='*80}")
        print(f"Baseline checkpoints: {self.baseline_checkpoints_dir}")
        print(f"Seeds: {self.seeds}")
        print(f"Bit widths: {self.bit_widths}")
        print(f"Lambdas: {self.lambdas}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}")
        
        # Store all results for final aggregation
        all_results = {
            'qat_results': defaultdict(lambda: defaultdict(lambda: {
                'fp32_accuracy': [],
                'int_accuracy': [],
                'mse': []
            }))
        }
        
        # Loop over seeds
        for seed_idx, seed in enumerate(self.seeds):
            print(f"\n\n{'#'*80}")
            print(f"# SEED {seed} ({seed_idx+1}/{len(self.seeds)})")
            print(f"{'#'*80}")
            
            # Find baseline checkpoint
            baseline_checkpoint = self.find_baseline_checkpoint(seed)
            if not baseline_checkpoint:
                print(f"✗ Error: Could not find baseline checkpoint for seed {seed}")
                continue
            
            print(f"✓ Found baseline: {baseline_checkpoint}")
            
            # Fine-tune with QAT+BR for all combinations
            for num_bits in self.bit_widths:
                for lambda_br in self.lambdas:
                    try:
                        # Train QAT+BR
                        qat_checkpoint, best_acc = self.train_qat_br(
                            baseline_checkpoint,
                            num_bits,
                            lambda_br,
                            seed,
                            args
                        )
                        
                        # Evaluate
                        fp32_acc, int_acc, mse = self.evaluate_qat_model(qat_checkpoint, seed)
                        
                        # Store results
                        all_results['qat_results'][num_bits][lambda_br]['fp32_accuracy'].append(fp32_acc)
                        all_results['qat_results'][num_bits][lambda_br]['int_accuracy'].append(int_acc)
                        all_results['qat_results'][num_bits][lambda_br]['mse'].append(mse)
                        
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                        import traceback
                        traceback.print_exc()
                        # Store NaN for failed runs
                        all_results['qat_results'][num_bits][lambda_br]['fp32_accuracy'].append(np.nan)
                        all_results['qat_results'][num_bits][lambda_br]['int_accuracy'].append(np.nan)
                        all_results['qat_results'][num_bits][lambda_br]['mse'].append(np.nan)
            
            # Save intermediate results after each seed
            self._save_intermediate_results(all_results, seed)
        
        # Generate final aggregated results and tables
        self._generate_final_report(all_results)
    
    def _save_intermediate_results(self, all_results: Dict, current_seed: int):
        """Save intermediate results after each seed."""
        results_file = self.results_dir / f'intermediate_seed{current_seed}.json'
        
        # Convert defaultdict to regular dict for JSON serialization
        results_for_json = {
            'qat_results': {
                str(bits): {
                    str(lam): {
                        'fp32_accuracy': data['fp32_accuracy'],
                        'int_accuracy': data['int_accuracy'],
                        'mse': data['mse']
                    }
                    for lam, data in lams.items()
                }
                for bits, lams in all_results['qat_results'].items()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        print(f"\n✓ Intermediate results saved to: {results_file}")
    
    def _generate_final_report(self, all_results: Dict):
        """Generate final aggregated report with tables."""
        print(f"\n\n{'='*80}")
        print(f"FINAL AGGREGATED RESULTS")
        print(f"{'='*80}")
        
        # Create formatted tables per bit width
        for num_bits in sorted(self.bit_widths):
            print(f"\n{'='*80}")
            print(f"BIT WIDTH: {num_bits} (INT{num_bits})")
            print(f"{'='*80}")
            
            table_data = []
            for lambda_br in sorted(self.lambdas):
                results = all_results['qat_results'][num_bits][lambda_br]
                
                # Filter out NaN values
                fp32_values = [x for x in results['fp32_accuracy'] if not np.isnan(x)]
                int_values = [x for x in results['int_accuracy'] if not np.isnan(x)]
                mse_values = [x for x in results['mse'] if not np.isnan(x)]
                
                if fp32_values:
                    fp32_mean = np.mean(fp32_values)
                    fp32_std = np.std(fp32_values, ddof=1) if len(fp32_values) > 1 else 0.0
                else:
                    fp32_mean = fp32_std = np.nan
                
                if int_values:
                    int_mean = np.mean(int_values)
                    int_std = np.std(int_values, ddof=1) if len(int_values) > 1 else 0.0
                else:
                    int_mean = int_std = np.nan
                
                if mse_values:
                    mse_mean = np.mean(mse_values)
                    mse_std = np.std(mse_values, ddof=1) if len(mse_values) > 1 else 0.0
                else:
                    mse_mean = mse_std = np.nan
                
                table_data.append({
                    'Lambda': lambda_br,
                    'FP32 Acc': f'{fp32_mean:.2f}±{fp32_std:.2f}',
                    f'INT{num_bits} Acc': f'{int_mean:.2f}±{int_std:.2f}',
                    'Avg. MSE': f'{mse_mean:.5f}±{mse_std:.5f}'
                })
            
            # Print table
            df = pd.DataFrame(table_data)
            print(df.to_string(index=False))
            
            # Save table as CSV
            csv_path = self.results_dir / f'results_table_{num_bits}bit.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Table saved to: {csv_path}")
        
        # Save combined summary
        summary_path = self.results_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"QAT+BR SWEEP SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Baseline checkpoints: {self.baseline_checkpoints_dir}\n")
            f.write(f"  Seeds: {self.seeds}\n")
            f.write(f"  Bit widths: {self.bit_widths}\n")
            f.write(f"  Lambdas: {self.lambdas}\n\n")
            
            for num_bits in sorted(self.bit_widths):
                f.write("="*80 + "\n")
                f.write(f"BIT WIDTH: {num_bits} (INT{num_bits})\n")
                f.write("="*80 + "\n\n")
                
                table_data = []
                for lambda_br in sorted(self.lambdas):
                    results = all_results['qat_results'][num_bits][lambda_br]
                    
                    fp32_values = [x for x in results['fp32_accuracy'] if not np.isnan(x)]
                    int_values = [x for x in results['int_accuracy'] if not np.isnan(x)]
                    mse_values = [x for x in results['mse'] if not np.isnan(x)]
                    
                    if fp32_values:
                        fp32_mean = np.mean(fp32_values)
                        fp32_std = np.std(fp32_values, ddof=1) if len(fp32_values) > 1 else 0.0
                    else:
                        fp32_mean = fp32_std = np.nan
                    
                    if int_values:
                        int_mean = np.mean(int_values)
                        int_std = np.std(int_values, ddof=1) if len(int_values) > 1 else 0.0
                    else:
                        int_mean = int_std = np.nan
                    
                    if mse_values:
                        mse_mean = np.mean(mse_values)
                        mse_std = np.std(mse_values, ddof=1) if len(mse_values) > 1 else 0.0
                    else:
                        mse_mean = mse_std = np.nan
                    
                    table_data.append({
                        'Lambda': lambda_br,
                        'FP32 Acc': f'{fp32_mean:.2f}±{fp32_std:.2f}',
                        f'INT{num_bits} Acc': f'{int_mean:.2f}±{int_std:.2f}',
                        'Avg. MSE': f'{mse_mean:.5f}±{mse_std:.5f}'
                    })
                
                df = pd.DataFrame(table_data)
                f.write(df.to_string(index=False))
                f.write("\n\n")
        
        print(f"✓ Summary saved to: {summary_path}")
        
        # Save raw results as JSON
        json_path = self.results_dir / 'raw_results.json'
        results_for_json = {
            'config': {
                'baseline_checkpoints_dir': str(self.baseline_checkpoints_dir),
                'seeds': self.seeds,
                'bit_widths': self.bit_widths,
                'lambdas': self.lambdas
            },
            'qat_results': {
                str(bits): {
                    str(lam): {
                        'fp32_accuracy': data['fp32_accuracy'],
                        'int_accuracy': data['int_accuracy'],
                        'mse': data['mse']
                    }
                    for lam, data in lams.items()
                }
                for bits, lams in all_results['qat_results'].items()
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        print(f"✓ Raw results saved to: {json_path}")
        
        print(f"\n{'='*80}")
        print(f"SWEEP COMPLETE!")
        print(f"{'='*80}")
        print(f"All results saved to: {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Automated QAT+BR Training + Evaluation Pipeline')
    
    # Input checkpoints
    parser.add_argument('--baseline-checkpoints-dir', type=str, required=True,
                       help='Directory containing baseline checkpoints from PTQ sweep')
    
    # Sweep parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='List of random seeds to use')
    parser.add_argument('--bit-widths', type=int, nargs='+', default=[1, 2, 4],
                       help='List of bit widths for quantization')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.1, 1.0, 10.0],
                       help='List of lambda values for BR regularization')
    
    # QAT+BR training parameters
    parser.add_argument('--qat-epochs', type=int, default=30, help='Number of QAT training epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs (LSQ only, no BR)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--freeze-alpha', action='store_true', 
                       help='Freeze alpha after warmup')
    parser.add_argument('--br-backprop-to-alpha', action='store_true',
                       help='Allow BR loss to backprop into alpha')
    
    # Device and output
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints and results')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    
    # Create runner
    runner = QATBRSweepRunner(
        baseline_checkpoints_dir=args.baseline_checkpoints_dir,
        seeds=args.seeds,
        bit_widths=args.bit_widths,
        lambdas=args.lambdas,
        output_dir=args.output_dir,
        device=device
    )
    
    # Run sweep
    runner.run_full_sweep(args)


if __name__ == '__main__':
    main()

