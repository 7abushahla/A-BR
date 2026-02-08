# Activation Distribution Visualization Guide

## Overview

The `compare_activations.py` script visualizes activation distributions from any checkpoint and overlays quantization levels. You run it multiple times with different checkpoints to compare them manually.

## Features

- **Auto-detects** model type (Baseline FP32 vs BR-trained QAT)
- **Extracts learned alphas** from BR models automatically
- **PTQ calibration** with configurable percentile for baseline models
- **Individual plots per ReLU layer** for detailed analysis
- **Supports both MNIST and CIFAR-10**
- **Works with all clip values** (ReLU, ReLU1, ReLU6)

---

## Usage Examples

### 1. Baseline Model (MNIST, ReLU1, 2-bit)

```bash
python experiments/compare_activations.py \
    --checkpoint results/mnist_relu1_sweep/checkpoints/mnist_resnet18_clip1.0_seed42_*.pth \
    --dataset mnist \
    --num-bits 2 \
    --calibration-percentile 99.9 \
    --output-dir plots/mnist_baseline_relu1_2bit_seed42/
```

**Output:** Individual histogram plots for each ReLU layer showing:
- Activation distribution (blue histogram)
- PTQ quantization levels (red dashed lines) computed via 99.9 percentile calibration

---

### 2. BR-Trained Model (MNIST, ReLU1, 2-bit, λ=1.0)

```bash
python experiments/compare_activations.py \
    --checkpoint results/mnist_relu1_br_sweep/checkpoints/qat_br_seed42_bits2_lambda1.0_*.pth \
    --dataset mnist \
    --num-bits 2 \
    --output-dir plots/mnist_br_relu1_2bit_lambda1_seed42/
```

**Output:** Individual histogram plots showing:
- Activation distribution shaped by BR training
- **Learned quantization levels** (red dashed lines) extracted from the model's `alpha` parameters
- Should show tighter clustering around quantization bins

---

### 3. CIFAR-10 Baseline (ReLU6, 4-bit)

```bash
python experiments/compare_activations.py \
    --checkpoint results/relu6_sweep/checkpoints/cifar10_resnet18_clip6.0_seed42_20260107_162512.pth \
    --dataset cifar10 \
    --num-bits 4 \
    --calibration-percentile 100.0 \
    --output-dir plots/cifar10_baseline_relu6_4bit_seed42/
```

---

### 4. CIFAR-10 BR Model (ReLU6, 1-bit, λ=0.1)

```bash
python experiments/compare_activations.py \
    --checkpoint results/relu6_br_sweep_seed42/checkpoints/qat_br_clip6.0_seed42_b1_lam10.0_20260107_205423.pth \
    --dataset cifar10 \
    --num-bits 1 \
    --output-dir plots/cifar10_br_relu6_1bit_lambda10_seed42/
```
---

python experiments/compare_activations.py \
    --checkpoint results/relu1_sweep/checkpoints/cifar10_resnet18_clip1.0_seed42_*.pth \
    --dataset cifar10 \
    --num-bits 1 \
    --calibration-percentile 100.0 \
    --use-train-data --train-mode \
    --output-dir plots/cifar10_baseline_relu1_1bit_seed42/



python experiments/compare_activations.py \
    --checkpoint results/relu1_br_sweep_seed42/checkpoints/qat_br_clip1.0_seed42_b1_lam10.0_*.pth \
    --dataset cifar10 --num-bits 1 \
    --use-train-data --train-mode \
    --plot-top-n 5 \
    --output-dir plots/report/cifar10_relu1_1bit_lam10_TRAIN/



## Comparing Multiple Configurations

### Example Workflow: Compare Baseline vs BR for ReLU1, 2-bit

```bash
# 1. Generate baseline plots
python experiments/compare_activations.py \
    --checkpoint results/mnist_relu1_sweep/checkpoints/mnist_resnet18_clip1.0_seed42_*.pth \
    --dataset mnist --num-bits 2 --calibration-percentile 99.9 \
    --output-dir plots/comparison/baseline_relu1_2bit/

# 2. Generate BR plots with λ=0.1
python experiments/compare_activations.py \
    --checkpoint results/mnist_relu1_br_sweep/checkpoints/qat_br_seed42_bits2_lambda0.1_*.pth \
    --dataset mnist --num-bits 2 \
    --output-dir plots/comparison/br_relu1_2bit_lambda0.1/

# 3. Generate BR plots with λ=1.0
python experiments/compare_activations.py \
    --checkpoint results/mnist_relu1_br_sweep/checkpoints/qat_br_seed42_bits2_lambda1.0_*.pth \
    --dataset mnist --num-bits 2 \
    --output-dir plots/comparison/br_relu1_2bit_lambda1.0/

# 4. Generate BR plots with λ=10.0
python experiments/compare_activations.py \
    --checkpoint results/mnist_relu1_br_sweep/checkpoints/qat_br_seed42_bits2_lambda10.0_*.pth \
    --dataset mnist --num-bits 2 \
    --output-dir plots/comparison/br_relu1_2bit_lambda10.0/
```

Now you can manually compare the histograms across these 4 directories to see how BR with different λ values affects the alignment with quantization levels.

---

## Expected Visual Insights

### Baseline PTQ Model:
- **Wide distribution** of activations
- Activations **not aligned** with quantization levels
- Quantization levels derived from calibration (percentile-based)

### BR-Trained Model:
- **Tighter clustering** around quantization levels
- Activations **pushed toward** red lines (quantization bins)
- More pronounced effect with higher λ values
- Quantization levels are **learned** (alpha parameters)

### Effect of Clipping:
- **ReLU1**: All activations ≤ 1.0
- **ReLU6**: All activations ≤ 6.0
- **Standard ReLU**: Unbounded positive activations

---

## All Command-Line Options

```
Required:
  --checkpoint PATH         Path to checkpoint (supports wildcards)
  --dataset {mnist,cifar10} Dataset to use
  --num-bits INT            Number of quantization bits
  --output-dir PATH         Directory to save plots

Optional:
  --calibration-percentile FLOAT   Percentile for PTQ (default: 100.0)
  --calibration-batches INT        Number of batches (default: 10)
  --batch-size INT                 Batch size (default: 128)
  --max-samples INT                Max samples per layer (default: 100000)
  --gpu INT                        GPU id (default: 0)
```

---

## Output Files

For each ReLU layer in the model, **THREE PNG files** are generated:

```
output-dir/
├── relu_histogram_zoomed.png           # Auto-zoom to quantization levels
├── relu_histogram_log.png              # Log scale (zoomed)
├── relu_histogram_fullrange.png        # Full range [0, clip_value] or [0, 15]
├── layer1.0.relu_histogram_zoomed.png
├── layer1.0.relu_histogram_log.png
├── layer1.0.relu_histogram_fullrange.png
├── layer1.1.relu_histogram_zoomed.png
├── layer1.1.relu_histogram_log.png
├── layer1.1.relu_histogram_fullrange.png
├── ...
└── layer4.1.relu_histogram_fullrange.png
```

Each plot contains:
- **Histogram** of activation values (blue bars)
- **Quantization levels** (red dashed vertical lines)
- **Scale value** and **sample count** in the title
- **Info box** with min/max activation values and x-axis range

### Three Versions Explained:

1. **`_zoomed.png`** (Auto-zoom): 
   - X-axis zoomed to `1.5 × max_quantization_level` or 95th percentile
   - Best for seeing clustering around quantization levels
   - Cuts off long tail outliers

2. **`_log.png`** (Log-scale, zoomed):
   - Y-axis in log scale, same x-axis as zoomed version
   - Reveals low-count bins that are invisible in linear scale
   - Useful for sparse/dead neurons in deep layers

3. **`_fullrange.png`** (Full activation range):
   - X-axis: `[0, clip_value]` for ReLU1/ReLU6, `[0, 15]` for standard ReLU
   - Shows the complete distribution including outliers
   - Good for understanding overall activation statistics

---

## Tips for Analysis

1. **Pick one seed** (e.g., 42) to reduce variability when comparing configurations
2. **Use consistent percentile** across runs (e.g., always 99.9)
3. **Compare layer-by-layer**: Look at the same layer across different checkpoints
4. **Watch for collapse**: If all activations are at 0 in BR model, training likely collapsed
5. **Lambda effects**: Higher λ → stronger clustering, but risk of collapse
6. **Bit-width effects**: Lower bits → fewer quantization levels → harder to align

---

## Batch Processing Example

Create a shell script to generate plots for many checkpoints:

```bash
#!/bin/bash
# generate_all_plots.sh

DATASET="mnist"
SEED=42

# Baseline models
for CLIP in "None" "1.0" "6.0"; do
    python experiments/compare_activations.py \
        --checkpoint results/${DATASET}_relu${CLIP}_sweep/checkpoints/*_seed${SEED}_*.pth \
        --dataset ${DATASET} --num-bits 2 --calibration-percentile 99.9 \
        --output-dir plots/baseline_clip${CLIP}_2bit/
done

# BR models (all λ)
for CLIP in "None" "1.0" "6.0"; do
    for LAMBDA in "0.1" "1.0" "10.0"; do
        python experiments/compare_activations.py \
            --checkpoint results/${DATASET}_relu${CLIP}_br_sweep/checkpoints/qat_br_seed${SEED}_bits2_lambda${LAMBDA}_*.pth \
            --dataset ${DATASET} --num-bits 2 \
            --output-dir plots/br_clip${CLIP}_2bit_lambda${LAMBDA}/
    done
done
```

Run with: `bash generate_all_plots.sh`

---

## Troubleshooting

**Error: "No checkpoint found matching"**
- Check that the path is correct
- Try using absolute paths
- Verify the checkpoint file exists

**Error: "ModuleNotFoundError"**
- Make sure you're in the correct conda environment
- Check that all imports are available

**Warning: "Multiple checkpoints found"**
- Wildcard matched multiple files
- Script uses the first match
- Be more specific in the path pattern

**Plots look empty or wrong**
- Check that the model loaded correctly (see console output)
- Verify dataset matches the model (MNIST vs CIFAR-10)
- Try increasing `--calibration-batches` or `--max-samples`

