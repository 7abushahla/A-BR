# Maximum Parallelization: 9 GPUs (ALL at once!)

Run all 9 combinations (3 activation types × 3 seeds) in parallel on 9 GPUs.

**Total Runtime: ~2.5 hours** (vs 7.5 hours for 3 GPUs, vs 22.5 hours sequential)

---

## 🚀 All 9 Commands

### **Terminal 1 (GPU 0): ReLU, Seed 42**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu_sweep/checkpoints \
    --seeds 42 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu_br_sweep_seed42
```

### **Terminal 2 (GPU 1): ReLU, Seed 43**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu_sweep/checkpoints \
    --seeds 43 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu_br_sweep_seed43
```

### **Terminal 3 (GPU 2): ReLU, Seed 44**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu_sweep/checkpoints \
    --seeds 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu_br_sweep_seed44
```

---

### **Terminal 4 (GPU 3): ReLU1, Seed 42**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu1_sweep/checkpoints \
    --seeds 42 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu1_br_sweep_seed42
```

### **Terminal 5 (GPU 4): ReLU1, Seed 43**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu1_sweep/checkpoints \
    --seeds 43 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu1_br_sweep_seed43
```

### **Terminal 6 (GPU 5): ReLU1, Seed 44**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu1_sweep/checkpoints \
    --seeds 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu1_br_sweep_seed44
```

---

### **Terminal 7 (GPU 6): ReLU6, Seed 42**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu6_sweep/checkpoints \
    --seeds 42 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu6_br_sweep_seed42
```

### **Terminal 8 (GPU 7): ReLU6, Seed 43**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu6_sweep/checkpoints \
    --seeds 43 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu6_br_sweep_seed43
```

### **Terminal 9 (GPU 8): ReLU6, Seed 44**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu6_sweep/checkpoints \
    --seeds 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu6_br_sweep_seed44
```

---

## 📊 After All Complete, Merge Results

### **Merge ReLU:**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/merge_qat_br_results.py \
    --results-dirs results/relu_br_sweep_seed42/results \
                   results/relu_br_sweep_seed43/results \
                   results/relu_br_sweep_seed44/results \
    --output-dir results/relu_br_sweep/results \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0
```

### **Merge ReLU1:**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/merge_qat_br_results.py \
    --results-dirs results/relu1_br_sweep_seed42/results \
                   results/relu1_br_sweep_seed43/results \
                   results/relu1_br_sweep_seed44/results \
    --output-dir results/relu1_br_sweep/results \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0
```

### **Merge ReLU6:**
```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/merge_qat_br_results.py \
    --results-dirs results/relu6_br_sweep_seed42/results \
                   results/relu6_br_sweep_seed43/results \
                   results/relu6_br_sweep_seed44/results \
    --output-dir results/relu6_br_sweep/results \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0
```

---

## ⏱️ Timeline

| Method | Total Runtime |
|--------|---------------|
| **9 GPUs (All at once)** | **~2.5 hours** ⚡ |
| 3 GPUs (Per activation type) | ~7.5 hours |
| Sequential (1 GPU) | ~22.5 hours |

**Speedup:** 9× faster than sequential! 🚀

---

## 📦 What Each GPU Does

Each GPU runs **9 QAT+BR trainings sequentially**:
- 3 bit widths × 3 lambda values = 9 combinations
- Each training: 50 epochs (~15 min)
- Total per GPU: ~2.5 hours

All 9 GPUs finish at approximately the same time.

---

## 🎯 GPU Assignment Map

| GPU | Activation | Seed | Output Dir |
|-----|------------|------|------------|
| 0   | ReLU       | 42   | `results/relu_br_sweep_seed42` |
| 1   | ReLU       | 43   | `results/relu_br_sweep_seed43` |
| 2   | ReLU       | 44   | `results/relu_br_sweep_seed44` |
| 3   | ReLU1      | 42   | `results/relu1_br_sweep_seed42` |
| 4   | ReLU1      | 43   | `results/relu1_br_sweep_seed43` |
| 5   | ReLU1      | 44   | `results/relu1_br_sweep_seed44` |
| 6   | ReLU6      | 42   | `results/relu6_br_sweep_seed42` |
| 7   | ReLU6      | 43   | `results/relu6_br_sweep_seed43` |
| 8   | ReLU6      | 44   | `results/relu6_br_sweep_seed44` |

---

## ✅ Final Results After Merge

```
results/
├── relu_br_sweep/
│   └── results/
│       ├── results_table_1bit.csv   # mean±std over seeds 42,43,44
│       ├── results_table_2bit.csv
│       ├── results_table_4bit.csv
│       └── summary.txt
├── relu1_br_sweep/
│   └── results/
│       ├── results_table_1bit.csv
│       ├── results_table_2bit.csv
│       ├── results_table_4bit.csv
│       └── summary.txt
└── relu6_br_sweep/
    └── results/
        ├── results_table_1bit.csv
        ├── results_table_2bit.csv
        ├── results_table_4bit.csv
        └── summary.txt
```

---

**This is the ABSOLUTE FASTEST way to complete all experiments! 🚀**

Requires: **9 GPUs available simultaneously**

