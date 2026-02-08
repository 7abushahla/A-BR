# Automated PTQ Sweep Pipeline

This directory contains scripts to automate the full training + PTQ (Post-Training Quantization) evaluation workflow across multiple seeds, bit widths, and calibration percentiles.

## Overview

The pipeline consists of three main scripts:

1. **`cifar10_automated_ptq_sweep.py`**: Main automation script that trains models and evaluates PTQ
2. **`run_all_activations.sh`**: Convenience script to run sweeps for all activation types (ReLU, ReLU1, ReLU6)
3. **`compare_activations.py`**: Compare results across different activation types

## Quick Start

### Option 1: Run All Activations at Once

```bash
cd /path/to/A-BR
./experiments/run_all_activations.sh
```

This will sequentially run PTQ sweeps for:
- Standard ReLU (no clipping)
- ReLU1 (clip-value 1.0)
- ReLU6 (clip-value 6.0)

Results will be saved to:
- `results/relu_sweep/`
- `results/relu1_sweep/`
- `results/relu6_sweep/`

### Option 2: Run Individual Activation Type

```bash
# ReLU6 example
python experiments/cifar10_automated_ptq_sweep.py \
    --clip-value 6.0 \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --percentiles 100.0 99.9 \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.02 \
    --gpu 0 \
    --calibration-batches 10 \
    --output-dir results/relu6_sweep
```

## What the Pipeline Does

For each activation configuration:

1. **Training Phase** (for each seed):
   - Trains a ResNet18 model on CIFAR-10
   - Saves the best checkpoint
   - Records FP32 test accuracy

2. **PTQ Evaluation Phase** (for each trained model):
   - For each bit width (1, 2, 4):
     - For each calibration percentile (100.0, 99.9):
       - Collects activations for calibration
       - Calibrates quantizers using percentile method
       - Applies PTQ and evaluates on test set
       - Records INT accuracy and average MSE

3. **Aggregation Phase**:
   - Computes mean ± std across all seeds
   - Generates formatted tables
   - Saves results in multiple formats (CSV, JSON, TXT)

## Output Structure

After running a sweep, you'll get:

```
results/relu6_sweep/
├── checkpoints/
│   ├── cifar10_resnet18_clip6.0_seed42_*.pth
│   ├── cifar10_resnet18_clip6.0_seed43_*.pth
│   └── cifar10_resnet18_clip6.0_seed44_*.pth
└── results/
    ├── raw_results.json          # Raw data (accuracy, MSE per seed)
    ├── results_table.csv          # Formatted table with mean±std
    ├── summary.txt                # Human-readable summary
    ├── intermediate_seed42.json   # Intermediate results after seed 42
    ├── intermediate_seed43.json   # Intermediate results after seed 43
    └── intermediate_seed44.json   # Intermediate results after seed 44
```

## Comparing Results

After running sweeps for multiple activation types, compare them:

```bash
python experiments/compare_activations.py \
    --results-dirs results/relu_sweep/results \
                   results/relu1_sweep/results \
                   results/relu6_sweep/results \
    --labels "ReLU" "ReLU1" "ReLU6" \
    --output-dir results/activation_comparison
```

This generates:
- `comparison_table.csv`: Side-by-side comparison table
- `comparison_p*.png`: Bar plots comparing accuracy and MSE
- `pivot_tables.txt`: Pivot tables for easy analysis
- `summary.txt`: Complete summary

## Expected Results Format

The final table will look like:

| Activation | Bit Width | Percentile | FP32 Acc      | INT Acc       | Avg. MSE         |
|------------|-----------|------------|---------------|---------------|------------------|
| ReLU       | 1         | 100.0      | 94.56±0.12    | 10.23±0.45    | 0.03625±0.00123  |
| ReLU       | 1         | 99.9       | 94.56±0.12    | 10.37±0.52    | 0.03478±0.00156  |
| ReLU       | 2         | 100.0      | 94.56±0.12    | 85.89±1.23    | 0.01895±0.00089  |
| ReLU       | 2         | 99.9       | 94.56±0.12    | 86.12±1.15    | 0.01823±0.00095  |
| ReLU       | 4         | 100.0      | 94.56±0.12    | 93.45±0.34    | 0.00456±0.00034  |
| ReLU       | 4         | 99.9       | 94.56±0.12    | 93.52±0.32    | 0.00442±0.00038  |
| ReLU1      | 1         | 100.0      | 95.28±0.08    | 12.45±0.67    | 0.03044±0.00145  |
| ...        | ...       | ...        | ...           | ...           | ...              |

## Configuration Options

### Main Script (`cifar10_automated_ptq_sweep.py`)

**Activation Configuration:**
- `--clip-value`: Clipping value (e.g., `1.0`, `6.0`, or `None` for standard ReLU)

**Sweep Parameters:**
- `--seeds`: List of random seeds (default: `42 43 44`)
- `--bit-widths`: Quantization bit widths (default: `1 2 4`)
- `--percentiles`: Calibration percentiles (default: `100.0 99.9`)

**Training Parameters:**
- `--epochs`: Number of epochs (default: 50)
- `--batch-size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 0.02)
- `--momentum`: SGD momentum (default: 0.9)
- `--weight-decay`: Weight decay (default: 5e-4)
- `--pretrained`: Use ImageNet pretrained weights

**PTQ Parameters:**
- `--calibration-batches`: Number of batches for calibration (default: 10)

**Device:**
- `--gpu`: GPU ID (default: 0, use -1 for CPU)

**Output:**
- `--output-dir`: Output directory (required)

### Shell Script (`run_all_activations.sh`)

Edit the configuration variables at the top of the script:
```bash
SEEDS="42 43 44"
BIT_WIDTHS="1 2 4"
PERCENTILES="100.0 99.9"
EPOCHS=50
BATCH_SIZE=256
LR=0.02
GPU=0
```

Uncomment `PRETRAINED="--pretrained"` to use ImageNet pretrained weights.

## Tips

1. **Start Small**: Test with one seed and fewer epochs first
   ```bash
   python experiments/cifar10_automated_ptq_sweep.py \
       --clip-value 6.0 \
       --seeds 42 \
       --bit-widths 4 \
       --percentiles 99.9 \
       --epochs 5 \
       --output-dir results/test_run
   ```

2. **Monitor Progress**: Each seed's intermediate results are saved, so you can check progress during long runs

3. **Resume Failed Runs**: If a run fails, you can manually run the PTQ evaluation script on saved checkpoints:
   ```bash
   python experiments/cifar10_evaluate_quantization.py \
       --baseline-model checkpoints/cifar10_resnet18_*.pth \
       --num-bits 4 \
       --mode ptq \
       --model-type resnet18 \
       --calibration-batches 10 \
       --calibration-percentile 99.9 \
       --seed 42
   ```

4. **Parallel Execution**: Run different activation types in parallel on different GPUs:
   ```bash
   # Terminal 1
   python experiments/cifar10_automated_ptq_sweep.py --clip-value None --gpu 0 ...
   
   # Terminal 2
   python experiments/cifar10_automated_ptq_sweep.py --clip-value 1.0 --gpu 1 ...
   
   # Terminal 3
   python experiments/cifar10_automated_ptq_sweep.py --clip-value 6.0 --gpu 2 ...
   ```

## Troubleshooting

**Out of Memory (OOM):**
- Reduce `--batch-size` (try 128 or 64)
- Reduce `--calibration-batches` (try 5)
- The script automatically limits samples per layer for ResNet18

**Slow Training:**
- Use `--pretrained` for faster convergence (can reduce epochs to 20-30)
- Increase `--batch-size` if GPU memory allows

**Reproducibility:**
- Seeds are set for training, data loading, and PTQ calibration sampling
- Results should be reproducible across runs with the same configuration

## Example Complete Workflow

```bash
# 1. Run sweeps for all activation types
./experiments/run_all_activations.sh

# 2. Wait for completion (this will take hours/days depending on configuration)

# 3. Compare results
python experiments/compare_activations.py \
    --results-dirs results/relu_sweep/results \
                   results/relu1_sweep/results \
                   results/relu6_sweep/results \
    --labels "ReLU" "ReLU1" "ReLU6" \
    --output-dir results/activation_comparison

# 4. View results
cat results/activation_comparison/summary.txt
```

## Files

- `cifar10_automated_ptq_sweep.py`: Main automation script
- `run_all_activations.sh`: Run all activation types
- `compare_activations.py`: Compare results across activations
- `cifar10_resnet18_baseline.py`: Training script (imported)
- `cifar10_evaluate_quantization.py`: PTQ evaluation script (imported)

## Citation

If you use this automation pipeline in your research, please cite the original papers for:
- ResNet: He et al., "Deep Residual Learning for Image Recognition"
- PTQ methods: Your relevant quantization papers

