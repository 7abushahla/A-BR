# Parallel Execution Commands - QAT+BR (CIFAR-10)

**IMPORTANT:** Run these AFTER completing the PTQ sweep to have baseline checkpoints available.

Run these three commands in separate terminals, each on a different GPU.

## Terminal 1: ReLU (Standard, no clipping) (GPU 0)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu_sweep/checkpoints \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu_br_sweep
```

## Terminal 2: ReLU1 (clip-value 1.0) (GPU 1)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu1_sweep/checkpoints \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 30 \
    --warmup-epochs 5 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu1_br_sweep
```

## Terminal 3: ReLU6 (clip-value 6.0) (GPU 2)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir results/relu6_sweep/checkpoints \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 30 \
    --warmup-epochs 5 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir results/relu6_br_sweep
```

---

## What This Does

For **each activation type** and **each seed**:
- Loads the baseline checkpoint from PTQ sweep
- Fine-tunes with QAT+BR for **9 combinations**:
  - 3 bit widths (1, 2, 4) × 3 lambda values (0.1, 1.0, 10.0)
- Evaluates each: FP32 Acc, INT Acc, MSE
- Aggregates over 3 seeds → mean ± std

**Total per activation type:** 3 seeds × 9 combinations = **27 QAT+BR runs**

## Expected Runtime

- **Per seed, per combination**: ~10-15 minutes (30 epochs QAT)
- **Per seed total**: 9 combinations × 12 min ≈ **~2 hours**
- **Per activation type**: 3 seeds × 2 hours ≈ **~6 hours**
- **All 3 in parallel**: **~6 hours**

## Output Structure

```
results/relu_qat_br_sweep/
├── checkpoints/
│   ├── qat_br_clipNone_seed42_b1_lam0.1_*.pth
│   ├── qat_br_clipNone_seed42_b1_lam1.0_*.pth
│   ├── qat_br_clipNone_seed42_b1_lam10.0_*.pth
│   ├── ... (27 total checkpoints per activation type)
└── results/
    ├── raw_results.json
    ├── results_table_1bit.csv
    ├── results_table_2bit.csv
    ├── results_table_4bit.csv
    ├── summary.txt
    └── intermediate_seed*.json
```

## Expected Results Format

### Per Bit Width Tables:

```
BIT WIDTH: 1 (INT1)
Lambda    FP32 Acc         INT1 Acc         Avg. MSE
0.1       94.56±0.12      45.23±2.34      0.02345±0.00123
1.0       94.56±0.12      48.67±1.89      0.02134±0.00098
10.0      94.56±0.12      52.34±1.56      0.01987±0.00087

BIT WIDTH: 2 (INT2)
Lambda    FP32 Acc         INT2 Acc         Avg. MSE
0.1       94.56±0.12      87.23±0.89      0.00876±0.00045
1.0       94.56±0.12      89.45±0.67      0.00734±0.00039
10.0      94.56±0.12      90.12±0.54      0.00689±0.00034

BIT WIDTH: 4 (INT4)
Lambda    FP32 Acc         INT4 Acc         Avg. MSE
0.1       94.56±0.12      93.89±0.23      0.00234±0.00012
1.0       94.56±0.12      94.12±0.19      0.00198±0.00009
10.0      94.56±0.12      94.23±0.17      0.00187±0.00008
```

## Flags Explanation

- `--freeze-alpha`: Freeze LSQ scale after warmup (prevents escaping BR)
- `--br-backprop-to-alpha`: Allow BR to influence alpha (paper-faithful)
- `--warmup-epochs 5`: Learn LSQ scales only (no BR) for first 5 epochs
- `--qat-epochs 30`: Total training epochs (5 warmup + 25 BR)

## Compare with PTQ Results

After completion, compare PTQ vs QAT+BR:

```bash
# PTQ results are in:
results/relu_sweep/results/results_table.csv

# QAT+BR results are in:
results/relu_qat_br_sweep/results/results_table_*bit.csv
```

Expected improvement: QAT+BR should have **higher INT accuracy** and **lower MSE** than PTQ, especially for low bit widths (1-2 bits).

## Prerequisites

✅ Must have completed PTQ sweep first:
- `results/relu_sweep/checkpoints/` must contain 3 checkpoints (seeds 42, 43, 44)
- `results/relu1_sweep/checkpoints/` must contain 3 checkpoints
- `results/relu6_sweep/checkpoints/` must contain 3 checkpoints

If you haven't run PTQ sweep yet, run `PARALLEL_COMMANDS.md` first!

