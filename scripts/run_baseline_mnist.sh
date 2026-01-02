#!/bin/bash

# Train baseline MNIST model (no quantization)
# This serves as the starting point for comparison

echo "========================================"
echo "Training MNIST Baseline (FP32)"
echo "========================================"

python experiments/mnist_baseline.py \
    --epochs 20 \
    --batch-size 256 \
    --lr 0.001 \
    --seed 42 \
    --gpu 0

echo ""
echo "========================================"
echo "Baseline training complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Train QAT with LSQ only:"
echo "     bash scripts/run_lsq_only_mnist.sh"
echo "  2. Train QAT with LSQ + BR:"
echo "     bash run_mnist_example.sh"
echo ""

