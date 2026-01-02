# A-BR Testing Guide

## Overview

This document describes the comprehensive test suite for the paper-faithful Bin Regularization implementation.

## Test Suite Summary

### ✅ All Tests Passing

1. **Unit Tests** (`test_br_correctness.py`) - Core BR mechanics
2. **Gradient Tests** (`test_br_gradients.py`) - Gradient flow verification
3. **End-to-End Tests** (`test_mnist_quick.sh`) - Real training validation

---

## 1. Core Correctness Tests

**File:** `tests/test_br_correctness.py`

**Purpose:** Verifies all paper-faithful implementation details with synthetic data.

**What it tests:**

### Test 1: Bin Assignment (LSQ Integer Code)
- ✅ Bins assigned via `round(clip(v/s, Qn, Qp))` (LSQ method)
- ✅ NOT nearest-level distance
- ✅ Handles PyTorch's "round half to even" correctly

### Test 2: Single-Element Bin Handling
- ✅ MSE computed for single-element bins
- ✅ Variance = 0 for single-element bins
- ✅ Matches paper's treatment of V_i ≤ 1

### Test 3: Multi-Element Bin Handling
- ✅ MSE computed as (mean - center)²
- ✅ Variance computed with `unbiased=False`
- ✅ Both terms contribute to loss

### Test 4: Quantization MSE Consistency
- ✅ Metrics use LSQ assignment (not nearest-level)
- ✅ Consistent with actual quantizer behavior
- ✅ Reflects real quantization error

### Test 5: Gradient Flow to Alpha (Coupled Mode)
- ✅ BR backprops to `alpha` when enabled
- ✅ "Simultaneous update" (paper) works
- ✅ Allows BR to influence step size

### Test 6: Clipping Behavior at Boundaries
- ✅ Values < 0 → bin 0
- ✅ Values > Qp*alpha → bin Qp
- ✅ Edge cases handled correctly

### Test 7: Effectiveness Metric
- ✅ 100% for perfect clustering
- ✅ Lower for dispersed activations
- ✅ Meaningful for tracking BR progress

**Run:**
```bash
cd A-BR
conda activate SDR
python tests/test_br_correctness.py
```

**Expected output:**
```
================================================================================
ALL TESTS PASSED! ✅
================================================================================

Implementation is paper-faithful:
  ✅ Bin assignment via LSQ integer code
  ✅ MSE computed for all non-empty bins
  ✅ Variance computed correctly (unbiased=False)
  ✅ Metrics use LSQ assignment
  ✅ Gradient flow to alpha works
  ✅ Boundary clipping correct
  ✅ Effectiveness metric sensible
```

---

## 2. Gradient Flow Tests

**File:** `tests/test_br_gradients.py`

**Purpose:** Verifies gradient behavior in coupled vs decoupled modes.

**What it tests:**
- ✅ **Decoupled mode:** BR → activations only (alpha.grad = None)
- ✅ **Coupled mode:** BR → activations + alpha (alpha.grad ≠ None)

**Run:**
```bash
cd A-BR
conda activate SDR
python tests/test_br_gradients.py
```

**Expected output:**
```
======================================================================
GRADIENT FLOW TEST: BR → Alpha
======================================================================
[DECOUPLED] alpha_br.grad: None
            ✅ BR does NOT backprop to alpha (decoupled)

[COUPLED]   alpha_br.grad: tensor(...)
            ✅ BR backprops to alpha (paper-faithful)
======================================================================
PASS: Gradient flow control works as expected
```

---

## 3. End-to-End MNIST Tests

**File:** `tests/test_mnist_quick.sh`

**Purpose:** Validates the full training pipeline with real data.

**What it tests:**
- ✅ Coupled mode (paper-faithful) training completes
- ✅ Decoupled mode training completes
- ✅ No runtime errors
- ✅ BR effectiveness improves during training
- ✅ Model achieves reasonable accuracy

**Run:**
```bash
cd A-BR
conda activate SDR
bash tests/test_mnist_quick.sh
```

**Expected results:**
- Training completes without errors
- Test accuracy > 96% after just 2 epochs (4-bit)
- BR effectiveness 60-80% (limited epochs)
- Both modes (coupled/decoupled) work

**Typical output (abbreviated):**
```
Test 1: Coupled mode (BR backprops to alpha) - 2 epochs
---------------------------------------------------------
Epoch 1/2 [WARMUP]: Acc: 96.42%
Epoch 2/2 [FINE-TUNE]: Acc: 97.28%, BR Effectiveness: 77.4%
✅ Test 1 PASSED

Test 2: Decoupled mode (BR doesn't affect alpha) - 2 epochs
-------------------------------------------------------------
Epoch 1/2 [WARMUP]: Acc: 96.42%
Epoch 2/2 [FINE-TUNE]: Acc: 97.46%, BR Effectiveness: 67.5%
✅ Test 2 PASSED

ALL END-TO-END TESTS PASSED! ✅
```

---

## 4. Full Training Runs (Optional)

### MNIST Full Experiment

**Run:**
```bash
cd A-BR
conda activate SDR
bash run_mnist_example.sh
```

**What to expect:**
- 30 warmup epochs (LSQ only)
- 70 fine-tune epochs (LSQ + BR)
- Test accuracy ~98% (2-bit) or ~99% (4-bit)
- BR effectiveness 80-95%
- Takes ~20 minutes (CPU) or ~5 minutes (GPU)

**Key metrics to monitor:**
1. **BR Effectiveness:** Should increase during fine-tune stage (target: >80%)
2. **Test Accuracy:** Should improve or stay stable when BR is added
3. **Quantization MSE:** Should decrease (activations clustering tighter)

### Coupled vs Decoupled Ablation

Compare paper-faithful (coupled) vs original (decoupled):

```bash
# Coupled (paper-faithful)
python experiments/mnist_qat_binreg.py \
    --num-bits 2 \
    --warmup-epochs 30 \
    --epochs 100 \
    --lambda-br 0.1 \
    --br-backprop-to-alpha \
    --seed 42

# Decoupled (original)
python experiments/mnist_qat_binreg.py \
    --num-bits 2 \
    --warmup-epochs 30 \
    --epochs 100 \
    --lambda-br 0.1 \
    --seed 42
```

**Compare:**
- Final test accuracy
- BR effectiveness
- Quantization MSE
- Training stability (loss curves in TensorBoard)

---

## Test Matrix

| Test | What | How Long | Pass Criteria |
|------|------|----------|---------------|
| `test_br_correctness.py` | Core BR logic | <1 min | All 7 tests pass |
| `test_br_gradients.py` | Gradient flow | <1 min | Both modes work |
| `test_mnist_quick.sh` | End-to-end | ~3-5 min | Both modes train, acc>96% |
| `run_mnist_example.sh` | Full training | ~20 min | Acc>98%, effectiveness>80% |

---

## Interpretation Guide

### BR Effectiveness Metric

```
Effectiveness = 100% * (1 - mean_distance / (alpha * 0.5))
```

- **>85%:** Excellent clustering, activations very close to levels
- **70-85%:** Good clustering, BR working as intended
- **50-70%:** Moderate clustering, may need higher λ or more epochs
- **<50%:** Weak clustering, increase λ or check for bugs

### Quantization MSE

```
quantization_mse = mean((continuous_acts - quantized_acts)²)
```

- Should **decrease** during fine-tune stage
- Lower is better (activations closer to quantized grid)
- Typical values: 0.0001 (good) to 0.001 (needs work) for normalized activations

### Loss Trajectory

**Expected:**
1. **Warmup:** Task loss decreases rapidly, no BR loss
2. **Fine-tune:** Small BR loss spike, then both losses decrease
3. **Convergence:** Both losses stabilize, effectiveness increases

**Red flags:**
- Task loss increases when BR is added → λ too high
- BR loss doesn't decrease → check gradient flow
- Effectiveness stuck <50% → increase λ or warmup epochs

---

## Quick Verification (After Changes)

Run this minimal test sequence to verify the implementation:

```bash
cd A-BR
conda activate SDR

# 1. Unit tests (~30s)
python tests/test_br_correctness.py

# 2. Gradient tests (~10s)
python tests/test_br_gradients.py

# 3. Quick MNIST (~3-5min)
bash tests/test_mnist_quick.sh
```

If all three pass → implementation is correct and functional.

---

## Summary

### What We've Verified

✅ **Core mechanics:** LSQ bin assignment, MSE+variance computation  
✅ **Gradient flow:** BR can backprop to alpha (paper-faithful mode)  
✅ **End-to-end:** Real training works, no runtime errors  
✅ **Both modes:** Coupled (paper) and decoupled (original) functional  

### What the Tests Confirm

1. **Paper-faithful implementation:**
   - Bins defined by LSQ integer code
   - Target levels = multiples of learned α
   - MSE for all non-empty bins (including size 1)
   - Variance with `unbiased=False`
   - Metrics consistent with LSQ

2. **Gradient control:**
   - `--br-backprop-to-alpha`: enables paper's "simultaneous update"
   - Default (decoupled): alpha updated only by task loss

3. **Practical viability:**
   - Training completes without errors
   - BR improves clustering (effectiveness metric)
   - Accuracy comparable or better than LSQ alone

### Next Steps

- Run full 100-epoch experiments to compare coupled vs decoupled
- Try different λ values (0.01, 0.1, 1.0) to find sweet spot
- Extend to CIFAR-10 or other datasets
- Compare 2-bit vs 4-bit quantization

