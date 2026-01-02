# Activation Quantization Evaluation Guide

## Goal

Prove that **uniform activation distributions** lead to **lower quantization error** (MSE) at low bit-widths compared to natural (heavy-tailed) distributions.

## Workflow

### 1. Train Models

**Baseline (Natural Distribution):**
```bash
python experiments/mnist_baseline.py --epochs 40
# Saves to: checkpoints/mnist_baseline_TIMESTAMP.pth
```

**A-KURE (Uniform Distribution):**
```bash
python experiments/mnist_wasserstein.py \
    --lambda-wass 10.0 \
    --epochs 50 \
    --num-samples -1 \
    --seed 42
# Saves to: checkpoints/mnist_wasserstein_TIMESTAMP.pth
```

---

### 2. Quantize Individual Model

```bash
python quantize_activations.py \
    --model checkpoints/baseline.pth \
    --model-type baseline \
    --bits 2 4 6 8 \
    --calib-samples 500 \
    --output results/baseline_quant.json
```

**Output:**
```
Quantizing at 2 bits
==========================================
Calibrating quantizers...
  relu1: SymmetricUniformQuantizer(bits=2, scale=1.234567, max_abs=3.7037)
  relu2: SymmetricUniformQuantizer(bits=2, scale=0.987654, max_abs=2.9630)
  ...

MSE Results:
  relu1: MSE = 0.123456
  relu2: MSE = 0.098765
  ...
  Average MSE: 0.111111
```

---

### 3. Compare Baseline vs A-KURE

```bash
python compare_quantization.py \
    --baseline checkpoints/mnist_baseline_TIMESTAMP.pth \
    --akure checkpoints/mnist_wasserstein_TIMESTAMP.pth \
    --bits 2 3 4 6 8 \
    --output-dir quantization_results
```

**Output:**
```
================================================================================
Quantization MSE Comparison
================================================================================
Bit-width    Baseline MSE         A-KURE MSE           Improvement    
--------------------------------------------------------------------------------
2-bit        0.234567             0.123456             +47.35%
4-bit        0.045678             0.023456             +48.65%
6-bit        0.009876             0.006789             +31.25%
8-bit        0.001234             0.000987             +20.02%
================================================================================

✅ Plot saved to: quantization_results/mse_comparison.png
```

---

## Expected Results

### Hypothesis

At **low bit-widths** (2-4 bits), uniform distributions use quantization bins more efficiently:

| Distribution | 2-bit Bins | Usage |
|--------------|------------|-------|
| **Heavy-tailed** (Baseline) | [-3, -1, 1, 3] | Bins at ±3 barely used |
| **Uniform** (A-KURE) | [-3, -1, 1, 3] | All bins used evenly ✓ |

### Result

**A-KURE should have 30-50% lower MSE at 2-4 bits!**

---

## Technical Details

### Symmetric Uniform Quantization

```python
# Parameters:
scale = max_abs / (2^(M-1) - 1)

# Quantize:
q = clip(round(x / scale), -2^(M-1), 2^(M-1) - 1)

# Dequantize:
x_hat = q * scale

# Error:
MSE = mean((x - x_hat)^2)
```

### Calibration

- **Dataset**: 500 samples from training set
- **Per-layer**: Each layer gets its own scale
- **Metric**: MSE (Mean Squared Error)

---

## Files

- `quantize_activations.py` - PTQ evaluation for single model
- `compare_quantization.py` - Compare baseline vs A-KURE
- `quantization_results/` - Output directory with JSON and plots

---

## Next Steps

1. ✅ Prove uniform → lower MSE at low bits
2. Test on larger models (ResNet, MobileNet)
3. Try asymmetric quantization (with zero-point)
4. Evaluate accuracy degradation (not just MSE)
5. Compare to other methods (BN-based, percentile calibration)

