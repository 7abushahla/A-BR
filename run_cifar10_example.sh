#!/bin/bash
#
# Quick CIFAR-10 QAT-BR Example
#
# This runs a single QAT-BR training on CIFAR-10 at 2-bit with proper two-stage training.
#

set -e

echo "========================================"
echo "CIFAR-10 QAT with Bin Regularization"
echo "========================================"
echo ""

# Configuration
BITS=2
WARMUP=30
TOTAL_EPOCHS=100
LAMBDA=10.0
CLIP=1.0
LR=0.001
BATCH_SIZE=256
GPU=0

echo "Configuration:"
echo "  Bits: $BITS"
echo "  Warmup epochs: $WARMUP"
echo "  Total epochs: $TOTAL_EPOCHS"
echo "  Lambda BR: $LAMBDA"
echo "  Clip value: $CLIP"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU"
echo ""

# Run QAT-BR training
python experiments/cifar10_qat_binreg.py \
    --num-bits $BITS \
    --warmup-epochs $WARMUP \
    --epochs $TOTAL_EPOCHS \
    --lambda-br $LAMBDA \
    --clip-value $CLIP \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --freeze-alpha \
    --br-all-layers \
    --gpu $GPU

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Check ./runs/ for TensorBoard logs"
echo "Check ./checkpoints/ for saved model"
echo ""
echo "To view results:"
echo "  tensorboard --logdir=./runs"

