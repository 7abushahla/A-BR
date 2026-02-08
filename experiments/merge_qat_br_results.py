#!/usr/bin/env python3
"""
Merge QAT+BR results from multiple seed runs.

Usage:
    python experiments/merge_qat_br_results.py \
        --results-dirs results/relu_br_sweep_seed42/results \
                       results/relu_br_sweep_seed43/results \
                       results/relu_br_sweep_seed44/results \
        --output-dir results/relu_br_sweep/results
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_raw_results(results_dir: Path) -> dict:
    """Load raw_results.json from a results directory."""
    json_path = results_dir / 'raw_results.json'
    with open(json_path, 'r') as f:
        return json.load(f)


def merge_results(results_dirs: list) -> dict:
    """Merge results from multiple seed runs."""
    merged = {
        'qat_results': defaultdict(lambda: defaultdict(lambda: {
            'fp32_accuracy': [],
            'int_accuracy': [],
            'mse': []
        }))
    }
    
    for results_dir in results_dirs:
        data = load_raw_results(Path(results_dir))
        
        for bit_str, bit_data in data['qat_results'].items():
            bit_width = int(bit_str)
            for lam_str, lam_data in bit_data.items():
                lam = float(lam_str)
                
                # Extend lists with values from this seed
                merged['qat_results'][bit_width][lam]['fp32_accuracy'].extend(lam_data['fp32_accuracy'])
                merged['qat_results'][bit_width][lam]['int_accuracy'].extend(lam_data['int_accuracy'])
                merged['qat_results'][bit_width][lam]['mse'].extend(lam_data['mse'])
    
    return merged


def generate_report(merged_results: dict, output_dir: Path, bit_widths: list, lambdas: list, seeds: list):
    """Generate aggregated report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MERGED QAT+BR RESULTS")
    print(f"{'='*80}")
    print(f"Seeds merged: {seeds}")
    
    # Create tables per bit width
    for num_bits in sorted(bit_widths):
        print(f"\n{'='*80}")
        print(f"BIT WIDTH: {num_bits} (INT{num_bits})")
        print(f"{'='*80}")
        
        table_data = []
        for lambda_br in sorted(lambdas):
            results = merged_results['qat_results'][num_bits][lambda_br]
            
            # Filter out NaN
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
        
        # Print and save table
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        
        csv_path = output_dir / f'results_table_{num_bits}bit.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Table saved to: {csv_path}")
    
    # Save summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MERGED QAT+BR RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Seeds merged: {seeds}\n")
        f.write(f"Bit widths: {bit_widths}\n")
        f.write(f"Lambdas: {lambdas}\n\n")
        
        for num_bits in sorted(bit_widths):
            f.write("="*80 + "\n")
            f.write(f"BIT WIDTH: {num_bits} (INT{num_bits})\n")
            f.write("="*80 + "\n\n")
            
            table_data = []
            for lambda_br in sorted(lambdas):
                results = merged_results['qat_results'][num_bits][lambda_br]
                
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
    
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Save merged raw results
    json_path = output_dir / 'raw_results.json'
    results_for_json = {
        'config': {
            'seeds': seeds,
            'bit_widths': bit_widths,
            'lambdas': lambdas
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
            for bits, lams in merged_results['qat_results'].items()
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"✓ Merged raw results saved to: {json_path}")
    
    print(f"\n{'='*80}")
    print(f"MERGE COMPLETE!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Merge QAT+BR results from multiple seed runs')
    parser.add_argument('--results-dirs', type=str, nargs='+', required=True,
                       help='List of results directories to merge (one per seed)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for merged results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='Seeds being merged (for documentation)')
    parser.add_argument('--bit-widths', type=int, nargs='+', default=[1, 2, 4],
                       help='Bit widths (for documentation)')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.1, 1.0, 10.0],
                       help='Lambda values (for documentation)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MERGING QAT+BR RESULTS")
    print("="*80)
    print(f"Input directories: {len(args.results_dirs)}")
    for d in args.results_dirs:
        print(f"  - {d}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Merge results
    merged = merge_results(args.results_dirs)
    
    # Generate report
    generate_report(
        merged, 
        Path(args.output_dir),
        args.bit_widths,
        args.lambdas,
        args.seeds
    )


if __name__ == '__main__':
    main()

