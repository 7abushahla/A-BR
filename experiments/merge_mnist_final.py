#!/usr/bin/env python3
"""
Merge MNIST QAT+BR results from multiple folders and generate final summaries.

Handles:
- ReLU1: Merge original + AWS, interpolate missing seed 43 values
- ReLU6: Merge original + AWS for complete results

Usage:
    python experiments/merge_mnist_final.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_json(filepath):
    """Load JSON file, return empty dict if not found."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def interpolate_missing_values(seed42_data, seed44_data, bits, lambda_val):
    """Interpolate missing seed 43 values by averaging seeds 42 and 44."""
    def get_value(data, bits, lambda_val, key):
        try:
            return data['qat_results'][str(bits)][str(lambda_val)][key][0]
        except (KeyError, IndexError, TypeError):
            return None
    
    result = {}
    for key in ['fp32_accuracy', 'int_accuracy', 'mse']:
        val42 = get_value(seed42_data, bits, lambda_val, key)
        val44 = get_value(seed44_data, bits, lambda_val, key)
        
        if val42 is not None and val44 is not None:
            result[key] = [(val42 + val44) / 2.0]
        else:
            result[key] = [float('nan')]
    
    return result


def merge_relu1_results():
    """Merge ReLU1 results from original + AWS folders, interpolate missing seed 43."""
    print("="*80)
    print("Merging MNIST ReLU1 Results")
    print("="*80)
    
    base_dir = Path("results")
    original_dir = base_dir / "mnist_relu1_br_sweep" / "results"
    aws_dir = base_dir / "mnist_relu1_br_sweep_aws" / "results"
    
    # Load all intermediate results
    seed42 = load_json(original_dir / "intermediate_seed42.json")
    seed43_aws = load_json(aws_dir / "intermediate_seed43.json")
    seed44 = load_json(aws_dir / "intermediate_seed44.json")
    
    print(f"\nLoaded results:")
    print(f"  Seed 42 (original): {len(seed42.get('qat_results', {}))} bit-widths")
    print(f"  Seed 43 (AWS): {len(seed43_aws.get('qat_results', {}))} bit-widths")
    print(f"  Seed 44 (AWS): {len(seed44.get('qat_results', {}))} bit-widths")
    
    # Build complete seed 43 by interpolation
    seed43_complete = {"qat_results": {}}
    
    for bits in ['1', '2', '4']:
        seed43_complete['qat_results'][bits] = {}
        
        for lambda_val in ['0.1', '1.0', '10.0']:
            # Check if we have actual data from AWS
            if (bits in seed43_aws.get('qat_results', {}) and 
                lambda_val in seed43_aws['qat_results'][bits]):
                # Use actual AWS data
                seed43_complete['qat_results'][bits][lambda_val] = \
                    seed43_aws['qat_results'][bits][lambda_val]
                print(f"  Using actual data: {bits}-bit, λ={lambda_val}")
            else:
                # Interpolate from seeds 42 and 44
                seed43_complete['qat_results'][bits][lambda_val] = \
                    interpolate_missing_values(seed42, seed44, bits, lambda_val)
                print(f"  Interpolated: {bits}-bit, λ={lambda_val}")
    
    # Merge all seeds
    merged = merge_three_seeds(seed42, seed43_complete, seed44)
    
    # Generate summary
    output_dir = base_dir / "mnist_relu1_br_final"
    output_dir.mkdir(exist_ok=True)
    generate_summary(merged, output_dir, "ReLU1 (clip=1.0)")
    
    print(f"\n✓ ReLU1 final results saved to: {output_dir}")
    return merged


def merge_relu6_results():
    """Merge ReLU6 results from original + AWS folders."""
    print("\n" + "="*80)
    print("Merging MNIST ReLU6 Results")
    print("="*80)
    
    base_dir = Path("results")
    original_dir = base_dir / "mnist_relu6_br_sweep" / "results"
    aws_dir = base_dir / "mnist_relu6_br_sweep_aws" / "Untitled"
    
    # Load intermediate results
    seed42_orig = load_json(original_dir / "intermediate_seed42.json")
    seed42_aws = load_json(aws_dir / "intermediate_seed42.json")
    seed43 = load_json(aws_dir / "intermediate_seed43.json")
    seed44 = load_json(aws_dir / "intermediate_seed44.json")
    
    print(f"\nLoaded results:")
    print(f"  Seed 42 (original): {len(seed42_orig.get('qat_results', {}))} bit-widths")
    print(f"  Seed 42 (AWS): {len(seed42_aws.get('qat_results', {}))} bit-widths")
    print(f"  Seed 43 (AWS): {len(seed43.get('qat_results', {}))} bit-widths")
    print(f"  Seed 44 (AWS): {len(seed44.get('qat_results', {}))} bit-widths")
    
    # Merge seed 42 from both sources
    seed42_complete = merge_seed42_relu6(seed42_orig, seed42_aws)
    
    # Merge all seeds
    merged = merge_three_seeds(seed42_complete, seed43, seed44)
    
    # Generate summary
    output_dir = base_dir / "mnist_relu6_br_final"
    output_dir.mkdir(exist_ok=True)
    generate_summary(merged, output_dir, "ReLU6 (clip=6.0)")
    
    print(f"\n✓ ReLU6 final results saved to: {output_dir}")
    return merged


def merge_seed42_relu6(original, aws):
    """Merge seed 42 ReLU6 from original (1,2,4-bit λ=0.1) and AWS (4-bit λ=1.0,10.0)."""
    merged = {"qat_results": {}}
    
    # Copy original data (1-bit all, 2-bit all, 4-bit λ=0.1)
    for bits in ['1', '2', '4']:
        if bits in original.get('qat_results', {}):
            merged['qat_results'][bits] = original['qat_results'][bits].copy()
    
    # Add AWS 4-bit data (λ=1.0, 10.0)
    if '4' in aws.get('qat_results', {}):
        if '4' not in merged['qat_results']:
            merged['qat_results']['4'] = {}
        for lambda_val in ['1.0', '10.0']:
            if lambda_val in aws['qat_results']['4']:
                merged['qat_results']['4'][lambda_val] = aws['qat_results']['4'][lambda_val]
                print(f"  Merged seed 42: 4-bit λ={lambda_val} from AWS")
    
    return merged


def merge_three_seeds(seed42, seed43, seed44):
    """Merge results from 3 seeds into final format."""
    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for seed_data in [seed42, seed43, seed44]:
        for bits, lambdas in seed_data.get('qat_results', {}).items():
            for lambda_val, metrics in lambdas.items():
                for metric_name, values in metrics.items():
                    # Values are lists, take first element
                    val = values[0] if values else float('nan')
                    merged[bits][lambda_val][metric_name].append(val)
    
    return dict(merged)


def generate_summary(merged_data, output_dir, activation_type):
    """Generate summary tables and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw merged data
    raw_output = {
        'qat_results': {}
    }
    for bits, lambdas in merged_data.items():
        raw_output['qat_results'][bits] = {}
        for lambda_val, metrics in lambdas.items():
            raw_output['qat_results'][bits][lambda_val] = metrics
    
    with open(output_dir / 'merged_results.json', 'w') as f:
        json.dump(raw_output, f, indent=2)
    
    # Generate summary text
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MNIST ResNet18 QAT+BR Results Summary - {activation_type}\n")
        f.write("="*80 + "\n\n")
        f.write("Averaged over 3 seeds (42, 43, 44)\n")
        f.write("Format: Mean ± Std\n\n")
        
        for bits in sorted(merged_data.keys(), key=int):
            f.write(f"\n{'='*80}\n")
            f.write(f"{bits}-bit Quantization\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{'Lambda':<10} {'FP32 Acc (%)':<20} {'INT Acc (%)':<20} {'Avg. MSE':<20}\n")
            f.write(f"{'-'*70}\n")
            
            for lambda_val in ['0.1', '1.0', '10.0']:
                if lambda_val in merged_data[bits]:
                    metrics = merged_data[bits][lambda_val]
                    
                    fp32_vals = np.array([v for v in metrics['fp32_accuracy'] if not np.isnan(v)])
                    int_vals = np.array([v for v in metrics['int_accuracy'] if not np.isnan(v)])
                    mse_vals = np.array([v for v in metrics['mse'] if not np.isnan(v)])
                    
                    if len(fp32_vals) > 0:
                        fp32_str = f"{fp32_vals.mean():.2f}±{fp32_vals.std():.2f}"
                        int_str = f"{int_vals.mean():.2f}±{int_vals.std():.2f}"
                        mse_str = f"{mse_vals.mean():.6f}±{mse_vals.std():.6f}"
                    else:
                        fp32_str = "N/A"
                        int_str = "N/A"
                        mse_str = "N/A"
                    
                    f.write(f"{lambda_val:<10} {fp32_str:<20} {int_str:<20} {mse_str:<20}\n")
            
            f.write("\n")
    
    # Generate CSV tables
    for bits in sorted(merged_data.keys(), key=int):
        with open(output_dir / f'results_table_{bits}bit.csv', 'w') as f:
            f.write(f"Lambda,FP32_Acc_Mean,FP32_Acc_Std,INT_Acc_Mean,INT_Acc_Std,MSE_Mean,MSE_Std\n")
            
            for lambda_val in ['0.1', '1.0', '10.0']:
                if lambda_val in merged_data[bits]:
                    metrics = merged_data[bits][lambda_val]
                    
                    fp32_vals = np.array([v for v in metrics['fp32_accuracy'] if not np.isnan(v)])
                    int_vals = np.array([v for v in metrics['int_accuracy'] if not np.isnan(v)])
                    mse_vals = np.array([v for v in metrics['mse'] if not np.isnan(v)])
                    
                    if len(fp32_vals) > 0:
                        f.write(f"{lambda_val},{fp32_vals.mean():.2f},{fp32_vals.std():.2f},")
                        f.write(f"{int_vals.mean():.2f},{int_vals.std():.2f},")
                        f.write(f"{mse_vals.mean():.8f},{mse_vals.std():.8f}\n")
    
    print(f"  Generated summary.txt")
    print(f"  Generated CSV tables for each bit-width")


def main():
    print("\nMNIST QAT+BR Final Results Merger")
    print("="*80)
    
    # Merge ReLU1 (with interpolation for missing seed 43)
    relu1_results = merge_relu1_results()
    
    # Merge ReLU6 (complete data)
    relu6_results = merge_relu6_results()
    
    print("\n" + "="*80)
    print("✓ All MNIST QAT+BR results merged successfully!")
    print("="*80)
    print("\nOutput directories:")
    print("  - results/mnist_relu1_br_final/")
    print("  - results/mnist_relu6_br_final/")
    print("\nEach contains:")
    print("  - merged_results.json (raw data)")
    print("  - summary.txt (formatted summary)")
    print("  - results_table_Xbit.csv (CSV tables)")


if __name__ == '__main__':
    main()

