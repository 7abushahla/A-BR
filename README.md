# A-BR: Activation Bin Regularization

Implementation of **Bin Regularization (BR)** for activation quantization, based on the ICCV 2021 paper:

> "Improving Low-Precision Network Quantization via Bin Regularization"  
> Tiantian Han, Dong Li, Ji Liu, Lu Tian, Yi Shan

## Overview

**Bin Regularization** encourages activations to cluster tightly at quantization levels, creating sharp (Dirac-like) distributions that minimize quantization error.

### Key Insight

- **LSQ** learns WHERE the quantization grid should be (via learned step size `s`)
- **BR** makes activations STICK to that grid (minimize within-bin variance)
- They co-evolve: LSQ adjusts grid position, BR shapes activation distribution

## Project Structure

```
A-BR/
├── abr/                          # Core library
│   ├── __init__.py              # Package exports
│   ├── regularizer_binreg.py    # Bin Regularization loss
│   ├── lsq_quantizer.py         # LSQ quantization for activations
│   └── hooks.py                 # Activation capture hooks
├── experiments/                  # Training scripts
│   ├── mnist_baseline.py        # MNIST baseline (FP32)
│   ├── mnist_qat_lsq_only.py    # MNIST with LSQ only (no BR)
│   ├── mnist_qat_binreg.py      # MNIST with LSQ + BR
│   ├── cifar10_mobilenet_baseline.py     # CIFAR-10 baseline
│   └── cifar10_mobilenet_qat_binreg.py   # CIFAR-10 with LSQ + BR
├── scripts/                      # Helper scripts
│   ├── run_baseline_mnist.sh    # Train MNIST baseline
│   ├── run_lsq_only_mnist.sh    # Train with LSQ only
│   ├── run_cifar10_baseline.sh  # Train CIFAR-10 baseline
│   └── run_cifar10_qat_binreg.sh # Train CIFAR-10 with BR
├── tests/                        # Verification tests
│   └── test_br_gradients.py     # Gradient flow sanity test ✅
├── docs/                         # Documentation
│   └── BR_IMPLEMENTATION_REVIEW.md  # Detailed implementation review
├── checkpoints/                  # Saved models (generated)
├── logs/                         # Training logs (generated)
├── runs/                         # TensorBoard logs (generated)
├── README.md                     # This file
└── run_mnist_example.sh          # Quick start script

```

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision tensorboard

# Or use existing conda environment
conda activate SDR  # (or your environment with PyTorch)
```

### Run MNIST Experiment

```bash
# Two-stage training (paper's S2 strategy)
# Stage 1: Warmup (30 epochs) - learn LSQ step size
# Stage 2: Joint (70 epochs) - LSQ + BR co-evolve

python experiments/mnist_qat_binreg.py \
  --num-bits 2 \            # 4 levels (2-bit)
  --warmup-epochs 30 \      # Stabilize step size first
  --epochs 100 \            # Total epochs
  --lambda-br 0.1           # BR loss weight
```

### Two Optimization Modes

The implementation supports two gradient flow modes:

#### Mode A: Decoupled (Default)

```bash
python experiments/mnist_qat_binreg.py \
  --num-bits 2 --warmup-epochs 30 --epochs 100 --lambda-br 0.1
```

- BR loss only affects activations (alpha detached via `.item()`)
- LSQ updates alpha via task loss
- More stable, easier to tune
- **Note:** This is a variant of the paper's approach

#### Mode B: Coupled (Paper-Faithful)

```bash
python experiments/mnist_qat_binreg.py \
  --num-bits 2 --warmup-epochs 30 --epochs 100 --lambda-br 0.1 \
  --br-backprop-to-alpha  # Enable gradient path BR → alpha
```

- BR loss CAN backprop into alpha/step size
- Both L_CE and L_BR contribute gradients to step size
- "Simultaneous update" as paper describes
- May be less stable early (warmup critical)

### Verify Gradient Flow

Run the sanity test to verify both modes work correctly:

```bash
conda activate SDR  # Or your PyTorch environment
python tests/test_br_gradients.py
```

Expected output:
- **Detached mode:** alpha.grad = None ✅
- **Coupled mode:** alpha.grad = nonzero ✅

## Implementation Details

### S2 Training Strategy (Paper's Recommendation)

```
Stage 1 (Warmup, ~30 epochs):
  - Optimize LSQ step size (s) alone
  - No BR loss, only task loss
  - Let s stabilize to data-driven optimal values

Stage 2 (Joint Training, remaining epochs):
  - Add BR loss: L = L_CE + λ·L_BR
  - Continue optimizing s (do NOT freeze)
  - LSQ and BR co-evolve
```

### Gradient Flow

**Total gradient to step size s:**
```
∂L/∂s = ∂L_CE/∂s + λ·∂L_BR/∂s  (coupled mode)
         ↑              ↑
    via quantizer   via BR MSE term (mean-to-center)
```

**Note:** Only the MSE term (mean-to-center) contributes ∂L_BR/∂s. The variance term cannot backprop to s (values selected via stop-grad argmin).

### Key Hyperparameters

- `--num-bits`: Target bit-width (e.g., 2 for 4 levels)
- `--warmup-epochs`: Stage 1 duration (paper uses ~30)
- `--lambda-br`: BR loss weight (tune as needed, start with 0.1)
- `--clip-value`: ReLU clipping value (default 1.0)
- `--br-backprop-to-alpha`: Enable coupled mode (paper-faithful)

## Important Caveats

1. **Weight vs Activation BR:** The paper studies weight regularization. We apply it to activations. Hyperparameters may need tuning.

2. **Discontinuous loss surface:** Even with stop-grad on bin assignment, bin membership flips discontinuously. Warmup + gentle λ ramping helps.

3. **Only MSE term updates s:** Variance term shapes activations but doesn't affect step size (values selected via stop-grad argmin).

## Experimental Pipeline

### Full Comparison (MNIST)

To run a complete comparison:

```bash
# 1. Train baseline (FP32, no quantization)
bash scripts/run_baseline_mnist.sh

# 2. Train with LSQ only (quantization, no BR)
bash scripts/run_lsq_only_mnist.sh

# 3. Train with LSQ + BR (our method)
bash run_mnist_example.sh decoupled

# 4. Optional: Try coupled mode (paper-faithful)
bash run_mnist_example.sh coupled
```

**Expected results:**
- Baseline: ~98-99% accuracy (FP32)
- LSQ only: ~97-98% accuracy (2-bit, natural distribution)
- LSQ + BR: ~97-98% accuracy (2-bit, binned distribution)
- **Key difference:** BR achieves **sharp clustering** (high effectiveness score)

### Full Comparison (CIFAR-10)

```bash
# 1. Train baseline (may take 2-3 hours)
bash scripts/run_cifar10_baseline.sh

# 2. Note the checkpoint path from step 1, then:
bash scripts/run_cifar10_qat_binreg.sh checkpoints/<baseline_checkpoint>.pth
```

## Usage Examples

### MNIST 2-bit Quantization

```bash
# Conservative (decoupled, stable)
python experiments/mnist_qat_binreg.py \
  --num-bits 2 --warmup-epochs 30 --epochs 100 --lambda-br 0.1 --seed 42

# Aggressive (coupled, paper-faithful)
python experiments/mnist_qat_binreg.py \
  --num-bits 2 --warmup-epochs 30 --epochs 100 --lambda-br 0.1 --seed 42 \
  --br-backprop-to-alpha
```

### CIFAR-10 with MobileNetV2

```bash
python experiments/cifar10_mobilenet_qat_binreg.py \
  --num-bits 2 --warmup-epochs 30 --epochs 200 --lambda-br 0.1
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=runs/
```

Key metrics to track:
- `binreg/effectiveness`: BR effectiveness score (0-100%, higher = better clustering)
- `binreg/quantization_mse`: Actual quantization error
- `quant_scales/`: Learned step sizes per layer
- `activations_pre_quant/`: Continuous activation distributions
- `activations_post_quant/`: Discrete quantized distributions

### CSV Logs

Training metrics are also saved to `logs/` as CSV files for easy analysis.

## Documentation

See `docs/BR_IMPLEMENTATION_REVIEW.md` for:
- Detailed implementation review
- Gradient flow analysis
- Common pitfalls and how to avoid them
- Verification checklist

## Citation

If you use this code, please cite the original BR paper:

```bibtex
@inproceedings{han2021improving,
  title={Improving Low-Precision Network Quantization via Bin Regularization},
  author={Han, Tiantian and Li, Dong and Liu, Ji and Tian, Lu and Shan, Yi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5241--5250},
  year={2021}
}
```

## Related Work

- **LSQ:** Esser et al., "Learned Step Size Quantization" (ICLR 2020)
- **PACT:** Choi et al., "PACT: Parameterized Clipping Activation for Quantized Neural Networks" (2018)

## License

See parent project for license information.

