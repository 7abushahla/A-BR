#!/bin/bash

# A-BR: MNIST Example with 2-bit Quantization
# 
# This runs the two-stage training strategy from the BR paper:
# 1. Stage 1 (warmup, 30 epochs): Learn LSQ step size
# 2. Stage 2 (joint, 70 epochs): LSQ + BR co-evolve
#
# Usage:
#   bash run_mnist_example.sh [decoupled|coupled]
#
# Modes:
#   - decoupled: Default, more stable (alpha detached)
#   - coupled: Paper-faithful (alpha with gradient)

MODE="${1:-decoupled}"  # Default to decoupled

echo "========================================"
echo "A-BR: MNIST 2-bit Quantization"
echo "Mode: $MODE"
echo "========================================"

# Common arguments
COMMON_ARGS="--num-bits 2 --warmup-epochs 30 --epochs 100 --lambda-br 0.1 --seed 42 --batch-size 256 --lr 0.001"

if [ "$MODE" = "coupled" ]; then
    echo "Running COUPLED mode (paper-faithful, BR backprops to alpha)"
    python experiments/mnist_qat_binreg.py $COMMON_ARGS --br-backprop-to-alpha
elif [ "$MODE" = "decoupled" ]; then
    echo "Running DECOUPLED mode (default, alpha detached)"
    python experiments/mnist_qat_binreg.py $COMMON_ARGS
else
    echo "Error: Unknown mode '$MODE'"
    echo "Usage: bash run_mnist_example.sh [decoupled|coupled]"
    exit 1
fi

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "View results:"
echo "  - TensorBoard: tensorboard --logdir=runs/"
echo "  - Logs: ls logs/"
echo "  - Checkpoints: ls checkpoints/"

