# Parallel Execution Commands

Run these three commands in separate terminals, each on a different GPU.

## Terminal 1: Standard ReLU (GPU 0)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_ptq_sweep.py \
    --clip-value None \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --percentiles 100.0 99.9 \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.02 \
    --gpu 0 \
    --calibration-batches 10 \
    --pretrained \
    --output-dir results/relu_sweep
```

## Terminal 2: ReLU1 (clip-value 1.0) (GPU 1)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_ptq_sweep.py \
    --clip-value 1.0 \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --percentiles 100.0 99.9 \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.02 \
    --gpu 1 \
    --calibration-batches 10 \
    --pretrained \
    --output-dir results/relu1_sweep
```

## Terminal 3: ReLU6 (clip-value 6.0) (GPU 2)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/cifar10_automated_ptq_sweep.py \
    --clip-value 6.0 \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --percentiles 100.0 99.9 \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.02 \
    --gpu 2 \
    --calibration-batches 10 \
    --pretrained \
    --output-dir results/relu6_sweep
```

---

## Notes

- Each command uses a different GPU (0, 1, 2)
- `--pretrained` flag is included for faster convergence with ImageNet weights
- Remove `--pretrained` if you want to train from scratch
- All three will run in parallel and finish at roughly the same time

## After All Complete

Compare results with:

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/compare_activations.py \
    --results-dirs results/relu_sweep/results \
                   results/relu1_sweep/results \
                   results/relu6_sweep/results \
    --labels "ReLU" "ReLU1" "ReLU6" \
    --output-dir results/activation_comparison
```

## Expected Runtime

With 3 seeds × 50 epochs × (1 training + 6 PTQ evaluations):
- **Per activation type**: ~2-4 hours (depending on GPU)
- **Total (all 3 in parallel)**: ~2-4 hours

## Monitor Progress

Each terminal will show:
- Training progress for each seed
- PTQ evaluation progress
- Intermediate results saved after each seed

