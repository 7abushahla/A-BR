# Parallel Execution Commands - MNIST

Run these three commands in separate terminals, each on a different GPU.

## Terminal 1: Standard ReLU (GPU 0)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/mnist_automated_ptq_sweep.py \
    --clip-value None \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --percentiles 100.0 99.9 \
    --epochs 30 \
    --batch-size 256 \
    --lr 0.1 \
    --calibration-batches 10 \
    --output-dir results/mnist_relu_sweep
```

## Terminal 2: ReLU1 (clip-value 1.0) (GPU 1)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/mnist_automated_ptq_sweep.py \
    --clip-value 1.0 \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --percentiles 100.0 99.9 \
    --epochs 30 \
    --batch-size 256 \
    --lr 0.1 \
    --calibration-batches 10 \
    --output-dir results/mnist_relu1_sweep
```

## Terminal 3: ReLU6 (clip-value 6.0) (GPU 2)

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/mnist_automated_ptq_sweep.py \
    --clip-value 6.0 \
    --seeds 42 43 44 \
    --bit-widths 1 2 4 \
    --percentiles 100.0 99.9 \
    --epochs 30 \
    --batch-size 256 \
    --lr 0.1 \
    --calibration-batches 10 \
    --output-dir results/mnist_relu6_sweep
```

---

## Differences from CIFAR-10

- **Epochs**: 20 (vs 50 for CIFAR-10) - MNIST trains faster
- **Learning Rate**: 0.1 (vs 0.02 for CIFAR-10) - MNIST typically uses higher LR
- **Dataset**: MNIST (28×28 grayscale) vs CIFAR-10 (32×32 RGB)
- **Expected FP32 Accuracy**: ~99%+ for MNIST vs ~94-95% for CIFAR-10
- **No `--pretrained` flag**: MNIST doesn't benefit as much from ImageNet pretraining (grayscale vs RGB)

## After All Complete

Compare results with:

```bash
cd "/Users/hamza/Library/CloudStorage/GoogleDrive-b00090279@alumni.aus.edu/My Drive/Quantization Research  Thesis/ActReg/A-BR"

python experiments/compare_activations.py \
    --results-dirs results/mnist_relu_sweep/results \
                   results/mnist_relu1_sweep/results \
                   results/mnist_relu6_sweep/results \
    --labels "ReLU" "ReLU1" "ReLU6" \
    --output-dir results/mnist_activation_comparison
```

## Expected Runtime

With 3 seeds × 20 epochs × (1 training + 6 PTQ evaluations):
- **Per activation type**: ~30-60 minutes (MNIST is faster than CIFAR-10)
- **Total (all 3 in parallel)**: ~30-60 minutes

## Quick Test Run (1 epoch)

Test everything works before the full run:

```bash
python experiments/mnist_automated_ptq_sweep.py \
    --clip-value None \
    --seeds 42 \
    --bit-widths 4 \
    --percentiles 99.9 \
    --epochs 1 \
    --batch-size 256 \
    --lr 0.1 \
    --gpu 0 \
    --calibration-batches 10 \
    --output-dir results/mnist_test_run
```

Expected runtime: ~3-5 minutes

## Key Changes from CIFAR-10

1. **Model Input**: 1 channel (grayscale) instead of 3 (RGB)
2. **Image Size**: 28×28 instead of 32×32
3. **Normalization**: `(0.1307,)` and `(0.3081,)` instead of CIFAR-10 values
4. **No Data Augmentation**: MNIST doesn't use RandomCrop/HorizontalFlip
5. **Faster Training**: MNIST converges much faster than CIFAR-10

