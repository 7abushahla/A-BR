#!/bin/bash
#
# Complete CIFAR-10 Experiment: Baseline vs QAT-BR
#
# This script runs the full experiment workflow:
# 1. Train baseline FP32 model
# 2. Train QAT-BR model (optionally warm-started from baseline)
# 3. Evaluate both: Baseline+PTQ vs QAT-BR
#

set -e

# Configuration
BITS=2
CLIP_BASELINE=  # Empty = None = standard ReLU (wide activation range)
CLIP_QAT=1.0    # Clipped ReLU for QAT-BR (narrow, shaped range)
WARMUP=30
TOTAL_EPOCHS=100
LAMBDA=10.0
LR=0.001
BATCH_SIZE=256
GPU=0

echo "========================================================================"
echo "COMPLETE CIFAR-10 EXPERIMENT: Baseline PTQ vs QAT-BR"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Quantization: ${BITS}-bit ($(python -c "print(2**$BITS)") levels)"
echo "  Baseline clip: ${CLIP_BASELINE:-None (standard ReLU)}"
echo "  QAT-BR clip: $CLIP_QAT"
echo "  Epochs: Baseline=100, QAT-BR=${TOTAL_EPOCHS} (warmup=${WARMUP})"
echo "  Lambda BR: $LAMBDA"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU"
echo ""
echo "========================================================================"
echo ""

# Step 1: Train Baseline FP32 Model
echo "========================================================================"
echo "STEP 1: Training Baseline FP32 Model"
echo "========================================================================"
echo ""

python experiments/cifar10_baseline.py \
    --epochs 100 \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --gpu $GPU
    # No --clip-value flag = standard ReLU (wide range, harder to quantize)

# Find the most recent baseline checkpoint
BASELINE_CKPT=$(ls -t checkpoints/cifar10_simple_baseline_*.pth 2>/dev/null | head -n 1)

if [ -z "$BASELINE_CKPT" ]; then
    echo "ERROR: No baseline checkpoint found!"
    exit 1
fi

echo ""
echo "✓ Baseline training complete: $BASELINE_CKPT"
echo ""

# Step 2: Train QAT-BR Model (warm-started from baseline)
echo "========================================================================"
echo "STEP 2: Training QAT-BR Model (warm-started from baseline)"
echo "========================================================================"
echo ""

python experiments/cifar10_qat_binreg.py \
    --num-bits $BITS \
    --warmup-epochs $WARMUP \
    --epochs $TOTAL_EPOCHS \
    --lambda-br $LAMBDA \
    --clip-value $CLIP_QAT \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --freeze-alpha \
    --br-all-layers \
    --pretrained-baseline "$BASELINE_CKPT" \
    --gpu $GPU

# Find the most recent QAT-BR checkpoint
QAT_CKPT=$(ls -t checkpoints/cifar10_qat_binreg_*.pth 2>/dev/null | head -n 1)

if [ -z "$QAT_CKPT" ]; then
    echo "ERROR: No QAT-BR checkpoint found!"
    exit 1
fi

echo ""
echo "✓ QAT-BR training complete: $QAT_CKPT"
echo ""

# Step 3: Evaluate and Compare
echo "========================================================================"
echo "STEP 3: Evaluating Quantization Quality (Baseline PTQ vs QAT-BR)"
echo "========================================================================"
echo ""

python experiments/cifar10_evaluate_quantization.py \
    --baseline-model "$BASELINE_CKPT" \
    --qat-model "$QAT_CKPT" \
    --num-bits $BITS \
    --calibration-percentile 100.0 \
    --output-dir "results/cifar10_quantization_comparison_${BITS}bit" \
    --gpu $GPU

echo "========================================================================"
echo "EXPERIMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Baseline checkpoint: $BASELINE_CKPT"
echo "  QAT-BR checkpoint:   $QAT_CKPT"
echo ""
echo "View training progress:"
echo "  tensorboard --logdir=./runs"
echo ""
echo "========================================================================"

