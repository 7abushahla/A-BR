# Automation Scripts Summary

## 📦 Created Files

### Main Automation Scripts

1. **`cifar10_automated_ptq_sweep.py`** (602 lines)
   - Automates PTQ evaluation for CIFAR-10
   - Trains FP32 models for multiple seeds
   - Evaluates PTQ for multiple bit widths & percentiles
   - Aggregates results with mean ± std

2. **`mnist_automated_ptq_sweep.py`** (663 lines)
   - Same as above but for MNIST dataset
   - Adapted for 28×28 grayscale images
   - Single-channel input, different normalization

3. **`cifar10_automated_qat_br_sweep.py`** (800+ lines)
   - Automates QAT+BR training & evaluation
   - Loads baseline checkpoints from PTQ sweep
   - Fine-tunes for multiple (bit_width, lambda) combinations
   - Evaluates and aggregates QAT+BR results

4. **`compare_activations.py`** (existing, can be used for comparisons)
   - Compares results across activation types
   - Generates comparison tables and plots

### Command Reference Files

5. **`PARALLEL_COMMANDS.md`**
   - Copy-paste commands for CIFAR-10 PTQ sweep
   - 3 parallel GPU commands (ReLU, ReLU1, ReLU6)

6. **`PARALLEL_COMMANDS_MNIST.md`**
   - Copy-paste commands for MNIST PTQ sweep
   - 3 parallel GPU commands (ReLU, ReLU1, ReLU6)

7. **`PARALLEL_COMMANDS_QAT_BR.md`**
   - Copy-paste commands for CIFAR-10 QAT+BR sweep
   - 3 parallel GPU commands (ReLU, ReLU1, ReLU6)

8. **`COMPLETE_WORKFLOW.md`**
   - End-to-end workflow documentation
   - Timeline estimates
   - Troubleshooting guide
   - Expected results format

9. **`README_PTQ_AUTOMATION.md`**
   - Detailed documentation for PTQ automation
   - Configuration options
   - Examples and tips

10. **`run_all_activations.sh`**
    - Shell script to run all 3 activation types sequentially
    - Can be modified for different configurations

11. **`AUTOMATION_SUMMARY.md`** (this file)
    - Quick reference for all created scripts

---

## 🎯 Quick Start Guide

### Step 1: PTQ Sweep (CIFAR-10)

```bash
# Terminal 1 (GPU 0) - ReLU
python experiments/cifar10_automated_ptq_sweep.py --clip-value None --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 50 --batch-size 256 --lr 0.02 --gpu 0 --calibration-batches 10 --pretrained --output-dir results/relu_sweep

# Terminal 2 (GPU 1) - ReLU1
python experiments/cifar10_automated_ptq_sweep.py --clip-value 1.0 --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 50 --batch-size 256 --lr 0.02 --gpu 1 --calibration-batches 10 --pretrained --output-dir results/relu1_sweep

# Terminal 3 (GPU 2) - ReLU6
python experiments/cifar10_automated_ptq_sweep.py --clip-value 6.0 --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 50 --batch-size 256 --lr 0.02 --gpu 2 --calibration-batches 10 --pretrained --output-dir results/relu6_sweep
```

**Runtime:** ~2-4 hours per activation type

### Step 2: QAT+BR Sweep (CIFAR-10)

```bash
# Terminal 1 (GPU 0) - ReLU
python experiments/cifar10_automated_qat_br_sweep.py --baseline-checkpoints-dir results/relu_sweep/checkpoints --seeds 42 43 44 --bit-widths 1 2 4 --lambdas 0.1 1.0 10.0 --qat-epochs 30 --warmup-epochs 5 --freeze-alpha --br-backprop-to-alpha --gpu 0 --output-dir results/relu_qat_br_sweep

# Terminal 2 (GPU 1) - ReLU1
python experiments/cifar10_automated_qat_br_sweep.py --baseline-checkpoints-dir results/relu1_sweep/checkpoints --seeds 42 43 44 --bit-widths 1 2 4 --lambdas 0.1 1.0 10.0 --qat-epochs 30 --warmup-epochs 5 --freeze-alpha --br-backprop-to-alpha --gpu 1 --output-dir results/relu1_qat_br_sweep

# Terminal 3 (GPU 2) - ReLU6
python experiments/cifar10_automated_qat_br_sweep.py --baseline-checkpoints-dir results/relu6_sweep/checkpoints --seeds 42 43 44 --bit-widths 1 2 4 --lambdas 0.1 1.0 10.0 --qat-epochs 30 --warmup-epochs 5 --freeze-alpha --br-backprop-to-alpha --gpu 2 --output-dir results/relu6_qat_br_sweep
```

**Runtime:** ~6 hours per activation type

### Step 3: Compare Results

```bash
# Compare PTQ across activation types
python experiments/compare_activations.py \
    --results-dirs results/relu_sweep/results results/relu1_sweep/results results/relu6_sweep/results \
    --labels "ReLU" "ReLU1" "ReLU6" \
    --output-dir results/ptq_activation_comparison
```

---

## 📊 Output Format

### PTQ Results Table

| Bit Width | Percentile | FP32 Acc      | INT Acc       | Avg. MSE         |
|-----------|------------|---------------|---------------|------------------|
| 1         | 100.0      | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 1         | 99.9       | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 2         | 100.0      | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 2         | 99.9       | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 4         | 100.0      | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 4         | 99.9       | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |

### QAT+BR Results Tables (per bit width)

**Bit Width: 1 (INT1)**
| Lambda | FP32 Acc      | INT1 Acc      | Avg. MSE         |
|--------|---------------|---------------|------------------|
| 0.1    | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 1.0    | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 10.0   | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |

**Bit Width: 2 (INT2)**
| Lambda | FP32 Acc      | INT2 Acc      | Avg. MSE         |
|--------|---------------|---------------|------------------|
| 0.1    | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 1.0    | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 10.0   | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |

**Bit Width: 4 (INT4)**
| Lambda | FP32 Acc      | INT4 Acc      | Avg. MSE         |
|--------|---------------|---------------|------------------|
| 0.1    | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 1.0    | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |
| 10.0   | XX.XX±Y.YY%  | XX.XX±Y.YY%  | X.XXXXX±Y.YYY   |

---

## 🔧 Configuration Options

### PTQ Automation

```python
--clip-value         # None, 1.0, 6.0 (ReLU type)
--seeds              # List of random seeds (default: 42 43 44)
--bit-widths         # List of quantization bits (default: 1 2 4)
--percentiles        # Calibration percentiles (default: 100.0 99.9)
--epochs             # Training epochs (CIFAR: 50, MNIST: 30)
--batch-size         # Batch size (default: 256)
--lr                 # Learning rate (CIFAR: 0.02, MNIST: 0.1)
--pretrained         # Use ImageNet pretrained weights (CIFAR only)
--calibration-batches # PTQ calibration batches (default: 10)
--gpu                # GPU ID (default: 0)
--output-dir         # Output directory (required)
```

### QAT+BR Automation

```python
--baseline-checkpoints-dir # Directory with PTQ checkpoints (required)
--seeds                    # List of random seeds (default: 42 43 44)
--bit-widths               # List of quantization bits (default: 1 2 4)
--lambdas                  # BR lambda values (default: 0.1 1.0 10.0)
--qat-epochs               # QAT training epochs (default: 30)
--warmup-epochs            # Warmup epochs (default: 5)
--batch-size               # Batch size (default: 128)
--lr                       # Learning rate (default: 0.001)
--freeze-alpha             # Freeze LSQ scale after warmup
--br-backprop-to-alpha     # Allow BR to affect LSQ scale
--gpu                      # GPU ID (default: 0)
--output-dir               # Output directory (required)
```

---

## 📈 Expected Timeline

### Full CIFAR-10 Experiment

| Stage | Description | Runtime (3 GPUs parallel) |
|-------|-------------|---------------------------|
| 1     | PTQ Sweep (3 activation types) | ~2-4 hours |
| 2     | QAT+BR Sweep (3 activation types) | ~6 hours |
| **Total** | | **~8-10 hours** |

### Full MNIST Experiment

| Stage | Description | Runtime (3 GPUs parallel) |
|-------|-------------|---------------------------|
| 1     | PTQ Sweep (3 activation types) | ~30-60 minutes |
| **Total** | | **~1 hour** |

---

## 📁 File Locations

All scripts are in:
```
/Users/hamza/.../ActReg/A-BR/experiments/
```

All results will be saved in:
```
/Users/hamza/.../ActReg/A-BR/results/
```

---

## ✅ Key Features

### PTQ Automation
- ✅ Multi-seed training (reproducibility)
- ✅ Multiple bit widths (1, 2, 4)
- ✅ Multiple calibration percentiles (100.0, 99.9)
- ✅ Automatic aggregation (mean ± std)
- ✅ Intermediate checkpoints (resume-able)
- ✅ Formatted tables (CSV, TXT, JSON)

### QAT+BR Automation
- ✅ Loads from PTQ baselines
- ✅ Multiple lambda values (0.1, 1.0, 10.0)
- ✅ Multiple bit widths (1, 2, 4)
- ✅ Warmup + BR training stages
- ✅ Alpha freezing option
- ✅ BR backprop to alpha option
- ✅ Automatic aggregation (mean ± std)
- ✅ Per-bit-width tables

---

## 🎓 Use Cases

1. **Compare PTQ vs QAT+BR**: Run both stages, compare results
2. **Tune Lambda**: Find best λ for each bit width
3. **Activation Type Study**: Compare ReLU vs ReLU1 vs ReLU6
4. **Reproducibility**: Multiple seeds ensure robust results
5. **Paper Results**: Generate publication-ready tables

---

**All scripts are ready to use! Check `COMPLETE_WORKFLOW.md` for the full workflow. 🚀**

