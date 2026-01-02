#!/bin/bash

# Train MNIST with LSQ quantization only (no BR)
# This serves as a fair comparison to BR - same quantization, no regularization

echo "========================================"
echo "Training MNIST with LSQ Only (2-bit)"
echo "========================================"

python experiments/mnist_qat_lsq_only.py \
    --num-bits 2 \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --seed 42 \
    --gpu 0

echo ""
echo "========================================"
echo "LSQ-only training complete!"
echo "========================================"
echo ""
echo "Compare with BR:"
echo "  bash run_mnist_example.sh decoupled"
echo ""

