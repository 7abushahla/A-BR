# Complete Automated Workflow: PTQ + QAT+BR Comparison

This document describes the complete workflow for comparing PTQ (Post-Training Quantization) with QAT+BR (Quantization-Aware Training with Bin Regularization) across multiple activation types.

## 📋 Overview

The workflow consists of two main stages:

1. **Stage 1: PTQ Baseline** - Train FP32 models, evaluate with PTQ
2. **Stage 2: QAT+BR** - Fine-tune FP32 models with QAT+BR

Each stage runs for 3 activation types (ReLU, ReLU1, ReLU6) with 3 seeds each.

---

## 🎯 Stage 1: PTQ Baseline

### What It Does
- Trains ResNet18 models on CIFAR-10/MNIST (FP32)
- Evaluates each with PTQ for bit widths 1, 2, 4
- Tests calibration percentiles 100.0 and 99.9
- Aggregates results over 3 seeds

### Run Commands

#### CIFAR-10 (3 parallel terminals):

**Terminal 1 (GPU 0):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/cifar10_automated_ptq_sweep.py --clip-value None --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 50 --batch-size 256 --lr 0.02 --gpu 0 --calibration-batches 10 --pretrained --output-dir results/relu_sweep
```

**Terminal 2 (GPU 1):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/cifar10_automated_ptq_sweep.py --clip-value 1.0 --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 50 --batch-size 256 --lr 0.02 --gpu 1 --calibration-batches 10 --pretrained --output-dir results/relu1_sweep
```

**Terminal 3 (GPU 2):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/cifar10_automated_ptq_sweep.py --clip-value 6.0 --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 50 --batch-size 256 --lr 0.02 --gpu 2 --calibration-batches 10 --pretrained --output-dir results/relu6_sweep
```

**Expected Runtime:** ~2-4 hours per activation type

#### MNIST (3 parallel terminals):

**Terminal 1 (GPU 0):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/mnist_automated_ptq_sweep.py --clip-value None --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 30 --batch-size 256 --lr 0.1 --calibration-batches 10 --output-dir results/mnist_relu_sweep
```

**Terminal 2 (GPU 1):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/mnist_automated_ptq_sweep.py --clip-value 1.0 --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 30 --batch-size 256 --lr 0.1 --calibration-batches 10 --output-dir results/mnist_relu1_sweep
```

**Terminal 3 (GPU 2):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/mnist_automated_ptq_sweep.py --clip-value 6.0 --seeds 42 43 44 --bit-widths 1 2 4 --percentiles 100.0 99.9 --epochs 30 --batch-size 256 --lr 0.1 --calibration-batches 10 --output-dir results/mnist_relu6_sweep
```

**Expected Runtime:** ~30-60 minutes per activation type

### Output
```
results/
├── relu_sweep/
│   ├── checkpoints/           # 3 FP32 models (seeds 42, 43, 44)
│   └── results/
│       ├── results_table.csv  # PTQ results
│       └── summary.txt
├── relu1_sweep/
│   └── ...
└── relu6_sweep/
    └── ...
```

---

## 🚀 Stage 2: QAT+BR Fine-tuning

### What It Does
- Loads FP32 checkpoints from Stage 1
- Fine-tunes with QAT+BR for:
  - Bit widths: 1, 2, 4
  - Lambda values: 0.1, 1.0, 10.0
- Evaluates: FP32 Acc, INT Acc, MSE
- Aggregates over 3 seeds

### Run Commands

#### CIFAR-10 (3 parallel terminals):

**Terminal 1 (GPU 0):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/cifar10_automated_qat_br_sweep.py --baseline-checkpoints-dir results/relu_sweep/checkpoints --seeds 42 43 44 --bit-widths 1 2 4 --lambdas 0.1 1.0 10.0 --qat-epochs 30 --warmup-epochs 5 --batch-size 128 --lr 0.001 --freeze-alpha --br-backprop-to-alpha --gpu 0 --output-dir results/relu_qat_br_sweep
```

**Terminal 2 (GPU 1):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/cifar10_automated_qat_br_sweep.py --baseline-checkpoints-dir results/relu1_sweep/checkpoints --seeds 42 43 44 --bit-widths 1 2 4 --lambdas 0.1 1.0 10.0 --qat-epochs 30 --warmup-epochs 5 --batch-size 128 --lr 0.001 --freeze-alpha --br-backprop-to-alpha --gpu 1 --output-dir results/relu1_qat_br_sweep
```

**Terminal 3 (GPU 2):**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"
python experiments/cifar10_automated_qat_br_sweep.py --baseline-checkpoints-dir results/relu6_sweep/checkpoints --seeds 42 43 44 --bit-widths 1 2 4 --lambdas 0.1 1.0 10.0 --qat-epochs 30 --warmup-epochs 5 --batch-size 128 --lr 0.001 --freeze-alpha --br-backprop-to-alpha --gpu 2 --output-dir results/relu6_qat_br_sweep
```

**Expected Runtime:** ~6 hours per activation type

### Output
```
results/
├── relu_qat_br_sweep/
│   ├── checkpoints/                    # 27 QAT models (3 seeds × 9 combinations)
│   └── results/
│       ├── results_table_1bit.csv      # QAT+BR results per bit width
│       ├── results_table_2bit.csv
│       ├── results_table_4bit.csv
│       └── summary.txt
├── relu1_qat_br_sweep/
│   └── ...
└── relu6_qat_br_sweep/
    └── ...
```

---

## 📊 Compare Results

### PTQ vs QAT+BR (per activation type)

**PTQ Results:**
```
results/relu_sweep/results/results_table.csv
```

**QAT+BR Results:**
```
results/relu_qat_br_sweep/results/results_table_*bit.csv
```

### Compare Across Activation Types

**For PTQ:**
```bash
python experiments/compare_activations.py \
    --results-dirs results/relu_sweep/results \
                   results/relu1_sweep/results \
                   results/relu6_sweep/results \
    --labels "ReLU" "ReLU1" "ReLU6" \
    --output-dir results/ptq_activation_comparison
```

**For QAT+BR:** Create similar comparison script or manually compare the tables.

---

## 📈 Expected Results

### Stage 1: PTQ Results (CIFAR-10)

| Activation | Bit Width | Percentile | FP32 Acc | INT Acc | Avg. MSE |
|------------|-----------|------------|----------|---------|----------|
| ReLU       | 4         | 99.9       | 94.5±0.1 | 93.2±0.3| 0.004±0.001 |
| ReLU1      | 4         | 99.9       | 95.2±0.1 | 94.1±0.2| 0.003±0.001 |
| ReLU6      | 4         | 99.9       | 94.8±0.1 | 93.8±0.2| 0.003±0.001 |

### Stage 2: QAT+BR Results (CIFAR-10)

| Activation | Bit Width | Lambda | FP32 Acc | INT Acc | Avg. MSE |
|------------|-----------|--------|----------|---------|----------|
| ReLU       | 4         | 1.0    | 94.5±0.1 | 94.1±0.2| 0.002±0.001 |
| ReLU1      | 4         | 1.0    | 95.2±0.1 | 94.8±0.2| 0.002±0.001 |
| ReLU6      | 4         | 1.0    | 94.8±0.1 | 94.5±0.2| 0.002±0.001 |

**Key Insight:** QAT+BR should show:
- ✅ Higher INT accuracy than PTQ
- ✅ Lower MSE than PTQ
- ✅ Especially strong gains for 1-2 bit quantization

---

## ⏱️ Timeline

### Quick Test (1 epoch)
- **PTQ:** ~20-30 minutes total
- **QAT+BR:** Not recommended (needs full training)

### Full Run (Recommended)
1. **Stage 1 (PTQ):**
   - CIFAR-10: ~2-4 hours (3 GPUs in parallel)
   - MNIST: ~30-60 minutes (3 GPUs in parallel)

2. **Stage 2 (QAT+BR):**
   - CIFAR-10: ~6 hours (3 GPUs in parallel)
   - MNIST: TBD (script can be adapted)

**Total for CIFAR-10:** ~8-10 hours

---

## 🗂️ File Structure Summary

```
A-BR/experiments/
├── cifar10_automated_ptq_sweep.py          # Stage 1: PTQ automation
├── mnist_automated_ptq_sweep.py            # Stage 1: PTQ (MNIST)
├── cifar10_automated_qat_br_sweep.py       # Stage 2: QAT+BR automation
├── compare_activations.py                  # Compare across activation types
├── PARALLEL_COMMANDS.md                    # Stage 1 commands (CIFAR-10)
├── PARALLEL_COMMANDS_MNIST.md              # Stage 1 commands (MNIST)
├── PARALLEL_COMMANDS_QAT_BR.md             # Stage 2 commands (CIFAR-10)
└── COMPLETE_WORKFLOW.md                    # This file

results/
├── relu_sweep/                             # PTQ: ReLU
├── relu1_sweep/                            # PTQ: ReLU1
├── relu6_sweep/                            # PTQ: ReLU6
├── mnist_relu_sweep/                       # PTQ: MNIST ReLU
├── mnist_relu1_sweep/                      # PTQ: MNIST ReLU1
├── mnist_relu6_sweep/                      # PTQ: MNIST ReLU6
├── relu_qat_br_sweep/                      # QAT+BR: ReLU
├── relu1_qat_br_sweep/                     # QAT+BR: ReLU1
├── relu6_qat_br_sweep/                     # QAT+BR: ReLU6
└── *_activation_comparison/                # Comparisons
```

---

## 🔧 Troubleshooting

### Issue: Baseline checkpoints not found
**Solution:** Make sure Stage 1 (PTQ) completed successfully. Check:
```bash
ls results/relu_sweep/checkpoints/
# Should show 3 .pth files (seeds 42, 43, 44)
```

### Issue: Out of Memory (OOM)
**Solution:**
- Reduce `--batch-size` (try 64 or 128)
- Reduce `--calibration-batches` (try 5)
- For QAT+BR, reduce `--batch-size` to 64

### Issue: Results look wrong
**Solution:**
- Check intermediate results: `results/*/results/intermediate_seed*.json`
- Verify seeds are consistent across stages
- Check that baseline checkpoints match activation type

---

## 📝 Citation

If you use this automation pipeline, please cite:
- Original ResNet paper
- LSQ (Learned Step-size Quantization)
- BR (Bin Regularization) paper

---

## ✅ Checklist

### Before Starting:
- [ ] Have 3 GPUs available (or adjust commands for fewer GPUs)
- [ ] Installed all dependencies (`abr` package, PyTorch, etc.)
- [ ] Have sufficient disk space (~5-10 GB for all checkpoints)

### Stage 1 (PTQ):
- [ ] Run CIFAR-10 PTQ sweep (3 activation types)
- [ ] Run MNIST PTQ sweep (3 activation types) [Optional]
- [ ] Verify checkpoints created: 3 per activation type
- [ ] Review PTQ results tables

### Stage 2 (QAT+BR):
- [ ] Run CIFAR-10 QAT+BR sweep (3 activation types)
- [ ] Verify checkpoints created: 27 per activation type
- [ ] Review QAT+BR results tables

### Analysis:
- [ ] Compare PTQ vs QAT+BR accuracy improvements
- [ ] Compare MSE reductions
- [ ] Identify best lambda values per bit width
- [ ] Compare across activation types

---

**Good luck with your experiments! 🚀**

