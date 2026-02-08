#!/bin/bash
# Run PTQ sweep for all activation types (ReLU, ReLU1, ReLU6)
# This script sequentially runs the automated PTQ sweep for each activation configuration

# Configuration
SEEDS="42 43 44"
BIT_WIDTHS="1 2 4"
PERCENTILES="100.0 99.9"
EPOCHS=50
BATCH_SIZE=256
LR=0.02
GPU=0
CALIBRATION_BATCHES=10

# Pretrained flag (uncomment to use ImageNet pretrained weights)
# PRETRAINED="--pretrained"
PRETRAINED=""

echo "=========================================="
echo "Running PTQ Sweep for All Activations"
echo "=========================================="
echo "Seeds: $SEEDS"
echo "Bit widths: $BIT_WIDTHS"
echo "Percentiles: $PERCENTILES"
echo "Epochs: $EPOCHS"
echo "=========================================="

# 1. Standard ReLU (no clipping)
echo ""
echo "=========================================="
echo "1/3: Standard ReLU (no clipping)"
echo "=========================================="
python experiments/cifar10_automated_ptq_sweep.py \
    --clip-value None \
    --seeds $SEEDS \
    --bit-widths $BIT_WIDTHS \
    --percentiles $PERCENTILES \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --gpu $GPU \
    --calibration-batches $CALIBRATION_BATCHES \
    --output-dir results/relu_sweep \
    $PRETRAINED

# 2. ReLU1 (clip at 1.0)
echo ""
echo "=========================================="
echo "2/3: ReLU1 (clip-value 1.0)"
echo "=========================================="
python experiments/cifar10_automated_ptq_sweep.py \
    --clip-value 1.0 \
    --seeds $SEEDS \
    --bit-widths $BIT_WIDTHS \
    --percentiles $PERCENTILES \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --gpu $GPU \
    --calibration-batches $CALIBRATION_BATCHES \
    --output-dir results/relu1_sweep \
    $PRETRAINED

# 3. ReLU6 (clip at 6.0)
echo ""
echo "=========================================="
echo "3/3: ReLU6 (clip-value 6.0)"
echo "=========================================="
python experiments/cifar10_automated_ptq_sweep.py \
    --clip-value 6.0 \
    --seeds $SEEDS \
    --bit-widths $BIT_WIDTHS \
    --percentiles $PERCENTILES \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --gpu $GPU \
    --calibration-batches $CALIBRATION_BATCHES \
    --output-dir results/relu6_sweep \
    $PRETRAINED

echo ""
echo "=========================================="
echo "ALL SWEEPS COMPLETE!"
echo "=========================================="
echo "Results:"
echo "  ReLU:  results/relu_sweep/results/"
echo "  ReLU1: results/relu1_sweep/results/"
echo "  ReLU6: results/relu6_sweep/results/"
echo "=========================================="

