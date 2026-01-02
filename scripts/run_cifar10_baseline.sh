#!/bin/bash
#
# Train CIFAR-10 MobileNetV2 Baseline (FP32)
#
# This trains the baseline model without quantization.
# Expected accuracy: ~92-93% on CIFAR-10

set -e

cd "$(dirname "$0")/.."  # Go to A-BR root directory

echo "======================================================================"
echo "CIFAR-10 MobileNetV2 Baseline Training (FP32)"
echo "======================================================================"
echo ""
echo "This will train a standard MobileNetV2 on CIFAR-10."
echo "Expected accuracy: ~92-93%"
echo "Training time: ~2-3 hours (100 epochs)"
echo ""
echo "======================================================================"
echo ""

python experiments/cifar10_mobilenet_baseline.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight-decay 4e-5 \
    --gpu 0 \
    --seed 42

echo ""
echo "======================================================================"
echo "Baseline training complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check TensorBoard for training curves and activation histograms"
echo "  2. Note the checkpoint path (you'll need it for QAT+BR)"
echo "  3. Run QAT+BR training: ./run_cifar10_qat_binreg.sh <checkpoint_path>"
echo "======================================================================"

