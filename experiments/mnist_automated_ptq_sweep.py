#!/usr/bin/env python3
"""
Automated Training + PTQ Evaluation Pipeline for MNIST ResNet18

This script automates the full workflow:
1. Train models for multiple seeds
2. Evaluate each model with PTQ across multiple bit widths and percentiles
3. Aggregate results (mean ± std across seeds)
4. Generate formatted tables

Usage:
    # ReLU6 (clip-value 6.0) with 3 seeds
    python experiments/mnist_automated_ptq_sweep.py \
        --clip-value 6.0 \
        --seeds 42 43 44 \
        --bit-widths 1 2 4 \
        --percentiles 100 99.9 \
        --epochs 20 \
        --batch-size 256 \
        --lr 0.1 \
        --gpu 0 \
        --output-dir results/mnist_relu6_sweep

    # ReLU (clip-value 1.0) with 3 seeds
    python experiments/mnist_automated_ptq_sweep.py \
        --clip-value 1.0 \
        --seeds 42 43 44 \
        --bit-widths 1 2 4 \
        --percentiles 100 99.9 \
        --epochs 20 \
        --batch-size 256 \
        --lr 0.1 \
        --gpu 0 \
        --output-dir results/mnist_relu1_sweep

    # Standard ReLU (no clipping) with 3 seeds
    python experiments/mnist_automated_ptq_sweep.py \
        --clip-value None \
        --seeds 42 43 44 \
        --bit-widths 1 2 4 \
        --percentiles 100 99.9 \
        --epochs 20 \
        --batch-size 256 \
        --lr 0.1 \
        --gpu 0 \
        --output-dir results/mnist_relu_sweep
"""

import os
import sys
import argparse
import subprocess
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

# Import necessary components from existing scripts
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Import quantization utilities
from experiments.cifar10_evaluate_quantization import (
    collect_activations_for_calibration,
    calibrate_ptq_quantizers,
    apply_ptq_quantization,
    compute_quantization_mse
)


class ClippedReLU(nn.Module):
    """ReLU with configurable clipping value."""
    def __init__(self, clip_value=6.0):
        super().__init__()
        self.clip_value = clip_value
    
    def forward(self, x):
        return torch.clamp(x, 0.0, self.clip_value)
    
    def __repr__(self):
        return f'ClippedReLU(clip_value={self.clip_value})'


def replace_relu_with_clipped(model, clip_value=6.0):
    """Replace all ReLU layers with ClippedReLU."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, ClippedReLU(clip_value=clip_value))
        else:
            replace_relu_with_clipped(module, clip_value)
    return model


def get_resnet18_mnist(num_classes=10, pretrained=False, clip_value=None):
    """
    Get ResNet18 adapted for MNIST (28x28 grayscale images).
    
    Args:
        num_classes: Number of output classes (default: 10)
        pretrained: Load ImageNet pretrained weights (default: False)
        clip_value: If not None, replace ReLU with ClippedReLU(clip_value)
    
    Returns:
        Modified ResNet18 model for MNIST
    """
    # Load ResNet18
    if pretrained:
        weights = 'IMAGENET1K_V1'
        print("  Loading ImageNet pretrained weights...")
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)
    
    # MNIST adaptations (28x28 grayscale input)
    # 1. Modify first conv layer: 3 channels -> 1 channel, smaller kernel
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove MaxPool (too aggressive for 28x28)
    model.maxpool = nn.Identity()
    
    # 3. Modify final FC layer: 1000 classes -> 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 4. Optionally replace ReLU with ClippedReLU
    if clip_value is not None:
        print(f"  Replacing ReLU with ClippedReLU(clip_value={clip_value})...")
        model = replace_relu_with_clipped(model, clip_value=clip_value)
    
    return model


class PTQSweepRunner:
    """
    Manages the full training + PTQ evaluation pipeline for MNIST.
    """
    
    def __init__(
        self,
        clip_value: Optional[float],
        seeds: List[int],
        bit_widths: List[int],
        percentiles: List[float],
        output_dir: str,
        device: str = 'cuda:0'
    ):
        self.clip_value = clip_value
        self.seeds = seeds
        self.bit_widths = bit_widths
        self.percentiles = percentiles
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Storage for results
        self.results = defaultdict(lambda: defaultdict(list))
    
    def find_checkpoint(self, seed: int) -> Optional[Tuple[str, float]]:
        """
        Find existing checkpoint for a given seed.
        
        Returns:
            (checkpoint_path, fp32_accuracy) if found, None otherwise
        """
        # Pattern: mnist_resnet18_clip{clip_value}_seed{seed}_*.pth
        clip_str = str(self.clip_value) if self.clip_value is not None else "None"
        pattern = f"mnist_resnet18_clip{clip_str}_seed{seed}_*.pth"
        
        matches = glob(str(self.checkpoints_dir / pattern))
        
        if matches:
            # Get the most recent checkpoint
            checkpoint_path = max(matches, key=lambda p: Path(p).stat().st_mtime)
            
            # Load checkpoint to get FP32 accuracy
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                fp32_acc = checkpoint.get('best_accuracy', checkpoint.get('test_accuracy', 0.0))
                return checkpoint_path, fp32_acc
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load checkpoint {checkpoint_path}: {e}")
                return None
        
        return None
        
    def train_model(self, seed: int, args: argparse.Namespace) -> Tuple[str, float]:
        """
        Train a model for a given seed.
        
        Returns:
            checkpoint_path: Path to saved checkpoint
            fp32_accuracy: FP32 test accuracy
        """
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL - Seed {seed}")
        print(f"{'='*80}")
        
        # Set random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create timestamp for unique checkpoint name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"mnist_resnet18_clip{self.clip_value}_seed{seed}_{timestamp}.pth"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        # Get data loaders - MNIST specific transforms
        # MNIST: 28x28 grayscale, normalize with mean=0.1307, std=0.3081
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=4, pin_memory=True)
        
        # Create model
        model = get_resnet18_mnist(
            num_classes=10,
            pretrained=args.pretrained,
            clip_value=self.clip_value
        ).to(self.device)
        
        print(f"Model: ResNet18 for MNIST (clip_value={self.clip_value})")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optimizer and scheduler
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        if args.pretrained:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        else:
            milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        
        # Training loop
        best_acc = 0.0
        for epoch in range(args.epochs):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # Test
            model.eval()
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
            
            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{args.epochs} (LR={current_lr:.6f}): "
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
                    'pretrained': args.pretrained,
                    'clip_value': self.clip_value,
                    'seed': seed,
                }, checkpoint_path)
        
        print(f"\n✓ Training complete! Best Test Accuracy: {best_acc:.2f}%")
        print(f"✓ Model saved to: {checkpoint_path}")
        
        return str(checkpoint_path), best_acc
    
    def evaluate_ptq(
        self,
        checkpoint_path: str,
        num_bits: int,
        percentile: float,
        seed: int,
        calibration_batches: int = 10
    ) -> Tuple[float, float]:
        """
        Evaluate a model with PTQ for a specific bit width and percentile.
        
        Returns:
            accuracy: INT accuracy
            avg_mse: Average MSE across layers
        """
        print(f"\n  PTQ Evaluation: {num_bits}-bit, percentile={percentile}")
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        loader_rng = torch.Generator(device='cpu').manual_seed(seed + 1)
        sample_rng = torch.Generator(device='cpu').manual_seed(seed + 2)
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = get_resnet18_mnist(
            num_classes=10,
            pretrained=checkpoint.get('pretrained', False),
            clip_value=checkpoint.get('clip_value', None)
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Data loaders
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
        calib_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=True,
            num_workers=0,
            generator=loader_rng
        )
        
        # Collect activations for calibration
        calib_acts = collect_activations_for_calibration(
            model, calib_loader, self.device,
            num_batches=calibration_batches,
            max_samples_per_layer=500000,
            rng=sample_rng
        )
        
        # Calibrate quantizers
        quantizers = calibrate_ptq_quantizers(
            calib_acts,
            num_bits,
            percentile=percentile,
            method='percentile',
            rng=sample_rng
        )
        
        # Apply PTQ and evaluate
        accuracy, quant_acts, orig_acts = apply_ptq_quantization(
            model, quantizers, test_loader, self.device, rng=sample_rng
        )
        
        # Compute MSE
        mse_dict = compute_quantization_mse(orig_acts, quant_acts)
        avg_mse = np.mean(list(mse_dict.values()))
        
        print(f"    ✓ Accuracy: {accuracy:.2f}%, Avg MSE: {avg_mse:.6f}")
        
        return accuracy, avg_mse
    
    def run_full_sweep(self, args: argparse.Namespace, eval_only: bool = False):
        """
        Run the full training + PTQ sweep for all seeds.
        
        Args:
            args: Command line arguments
            eval_only: If True, skip training and only run PTQ evaluation on existing checkpoints
        """
        print(f"\n{'='*80}")
        print(f"AUTOMATED PTQ SWEEP - MNIST")
        print(f"{'='*80}")
        print(f"Activation: ClippedReLU(clip_value={self.clip_value})")
        print(f"Seeds: {self.seeds}")
        print(f"Bit widths: {self.bit_widths}")
        print(f"Percentiles: {self.percentiles}")
        print(f"Output directory: {self.output_dir}")
        if eval_only:
            print(f"Mode: EVALUATION ONLY (using existing checkpoints)")
        print(f"{'='*80}")
        
        # Store all results for final aggregation
        all_results = {
            'fp32_accuracy': [],
            'ptq_results': defaultdict(lambda: defaultdict(lambda: {'accuracy': [], 'mse': []}))
        }
        
        # Loop over seeds
        for seed_idx, seed in enumerate(self.seeds):
            print(f"\n\n{'#'*80}")
            print(f"# SEED {seed} ({seed_idx+1}/{len(self.seeds)})")
            print(f"{'#'*80}")
            
            # Step 1: Train model or find existing checkpoint
            if eval_only:
                result = self.find_checkpoint(seed)
                if result is None:
                    print(f"  ✗ Error: No checkpoint found for seed {seed}")
                    print(f"  Skipping seed {seed}...")
                    continue
                checkpoint_path, fp32_acc = result
                print(f"  ✓ Found checkpoint: {checkpoint_path}")
                print(f"  ✓ FP32 Accuracy: {fp32_acc:.2f}%")
            else:
                checkpoint_path, fp32_acc = self.train_model(seed, args)
            
            all_results['fp32_accuracy'].append(fp32_acc)
            
            # Step 2: Evaluate with PTQ for all bit widths and percentiles
            print(f"\n{'='*80}")
            print(f"PTQ EVALUATION - Seed {seed}")
            print(f"{'='*80}")
            
            for num_bits in self.bit_widths:
                for percentile in self.percentiles:
                    try:
                        int_acc, avg_mse = self.evaluate_ptq(
                            checkpoint_path,
                            num_bits,
                            percentile,
                            seed,
                            calibration_batches=args.calibration_batches
                        )
                        
                        # Store results
                        all_results['ptq_results'][num_bits][percentile]['accuracy'].append(int_acc)
                        all_results['ptq_results'][num_bits][percentile]['mse'].append(avg_mse)
                        
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                        # Store NaN for failed runs
                        all_results['ptq_results'][num_bits][percentile]['accuracy'].append(np.nan)
                        all_results['ptq_results'][num_bits][percentile]['mse'].append(np.nan)
            
            # Save intermediate results after each seed
            self._save_intermediate_results(all_results, seed)
        
        # Generate final aggregated results and tables
        self._generate_final_report(all_results)
    
    def _save_intermediate_results(self, all_results: Dict, current_seed: int):
        """Save intermediate results after each seed."""
        results_file = self.results_dir / f'intermediate_seed{current_seed}.json'
        
        # Convert defaultdict to regular dict for JSON serialization
        results_for_json = {
            'fp32_accuracy': all_results['fp32_accuracy'],
            'ptq_results': {
                str(bits): {
                    str(perc): {
                        'accuracy': data['accuracy'],
                        'mse': data['mse']
                    }
                    for perc, data in percs.items()
                }
                for bits, percs in all_results['ptq_results'].items()
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
        
        # FP32 accuracy
        fp32_mean = np.mean(all_results['fp32_accuracy'])
        fp32_std = np.std(all_results['fp32_accuracy'], ddof=1) if len(all_results['fp32_accuracy']) > 1 else 0.0
        
        print(f"\nFP32 Accuracy: {fp32_mean:.2f}% ± {fp32_std:.2f}%")
        print(f"  Individual seeds: {[f'{x:.2f}%' for x in all_results['fp32_accuracy']]}")
        
        # PTQ results table
        print(f"\n{'='*80}")
        print(f"PTQ RESULTS TABLE (mean ± std over {len(self.seeds)} seeds)")
        print(f"{'='*80}")
        
        # Create formatted table
        table_data = []
        for num_bits in sorted(self.bit_widths):
            for percentile in sorted(self.percentiles, reverse=True):
                results = all_results['ptq_results'][num_bits][percentile]
                
                # Filter out NaN values
                acc_values = [x for x in results['accuracy'] if not np.isnan(x)]
                mse_values = [x for x in results['mse'] if not np.isnan(x)]
                
                if acc_values:
                    acc_mean = np.mean(acc_values)
                    acc_std = np.std(acc_values, ddof=1) if len(acc_values) > 1 else 0.0
                else:
                    acc_mean = acc_std = np.nan
                
                if mse_values:
                    mse_mean = np.mean(mse_values)
                    mse_std = np.std(mse_values, ddof=1) if len(mse_values) > 1 else 0.0
                else:
                    mse_mean = mse_std = np.nan
                
                table_data.append({
                    'Bit Width': num_bits,
                    'Percentile': percentile,
                    'FP32 Acc': f'{fp32_mean:.2f}±{fp32_std:.2f}',
                    'INT Acc': f'{acc_mean:.2f}±{acc_std:.2f}',
                    'Avg MSE': f'{mse_mean:.5f}±{mse_std:.5f}'
                })
        
        # Print table
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        
        # Save table as CSV
        csv_path = self.results_dir / 'results_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Table saved to: {csv_path}")
        
        # Save detailed summary
        summary_path = self.results_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"PTQ SWEEP SUMMARY - MNIST\n")
            f.write("="*80 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Activation: ClippedReLU(clip_value={self.clip_value})\n")
            f.write(f"  Seeds: {self.seeds}\n")
            f.write(f"  Bit widths: {self.bit_widths}\n")
            f.write(f"  Percentiles: {self.percentiles}\n\n")
            f.write(f"FP32 Accuracy: {fp32_mean:.2f}% ± {fp32_std:.2f}%\n")
            f.write(f"  Seeds: {[f'{x:.2f}%' for x in all_results['fp32_accuracy']]}\n\n")
            f.write("="*80 + "\n")
            f.write("PTQ RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Detailed per-seed results
            f.write("="*80 + "\n")
            f.write("DETAILED PER-SEED RESULTS\n")
            f.write("="*80 + "\n\n")
            for num_bits in sorted(self.bit_widths):
                f.write(f"\n{num_bits}-bit quantization:\n")
                for percentile in sorted(self.percentiles, reverse=True):
                    results = all_results['ptq_results'][num_bits][percentile]
                    f.write(f"  Percentile {percentile}:\n")
                    f.write(f"    Accuracies: {[f'{x:.2f}%' for x in results['accuracy']]}\n")
                    f.write(f"    MSEs: {[f'{x:.6f}' for x in results['mse']]}\n")
        
        print(f"✓ Summary saved to: {summary_path}")
        
        # Save raw results as JSON
        json_path = self.results_dir / 'raw_results.json'
        results_for_json = {
            'config': {
                'dataset': 'MNIST',
                'clip_value': self.clip_value,
                'seeds': self.seeds,
                'bit_widths': self.bit_widths,
                'percentiles': self.percentiles
            },
            'fp32_accuracy': all_results['fp32_accuracy'],
            'ptq_results': {
                str(bits): {
                    str(perc): {
                        'accuracy': data['accuracy'],
                        'mse': data['mse']
                    }
                    for perc, data in percs.items()
                }
                for bits, percs in all_results['ptq_results'].items()
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
    parser = argparse.ArgumentParser(description='Automated Training + PTQ Evaluation Pipeline for MNIST')
    
    # Activation configuration
    parser.add_argument('--clip-value', type=str, default='6.0',
                       help='Clipping value for ReLU (e.g., "1.0" for ReLU1, "6.0" for ReLU6, "None" for standard ReLU)')
    
    # Sweep parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='List of random seeds to use')
    parser.add_argument('--bit-widths', type=int, nargs='+', default=[1, 2, 4],
                       help='List of bit widths for quantization')
    parser.add_argument('--percentiles', type=float, nargs='+', default=[100.0, 99.9],
                       help='List of percentiles for PTQ calibration')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1 for MNIST)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    
    # PTQ parameters
    parser.add_argument('--calibration-batches', type=int, default=10,
                       help='Number of batches for PTQ calibration')
    
    # Evaluation mode
    parser.add_argument('--eval-only', action='store_true',
                       help='Skip training and only run PTQ evaluation on existing checkpoints')
    
    # Device and output
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for checkpoints and results')
    
    args = parser.parse_args()
    
    # Parse clip-value
    if args.clip_value.lower() == 'none':
        clip_value = None
    else:
        clip_value = float(args.clip_value)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    
    # Create runner
    runner = PTQSweepRunner(
        clip_value=clip_value,
        seeds=args.seeds,
        bit_widths=args.bit_widths,
        percentiles=args.percentiles,
        output_dir=args.output_dir,
        device=device
    )
    
    # Run sweep
    runner.run_full_sweep(args, eval_only=args.eval_only)


if __name__ == '__main__':
    main()

