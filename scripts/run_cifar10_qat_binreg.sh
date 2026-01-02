#!/bin/bash
#
# CIFAR-10 MobileNetV2 with QAT + Bin Regularization
#
# Fine-tune the baseline model with 2-bit quantization + BR.
# Expected: 75-85% BR effectiveness (with BatchNorm)

set -e

cd "$(dirname "$0")/.."  # Go to A-BR root directory

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Error: Please provide path to baseline checkpoint"
    echo ""
    echo "Usage: $0 <baseline_checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  $0 checkpoints/cifar10_mobilenet_baseline_20260102_123456.pth"
    echo ""
    exit 1
fi

BASELINE_CHECKPOINT=$1

echo "======================================================================"
echo "CIFAR-10 MobileNetV2 with QAT + Bin Regularization"
echo "======================================================================"
echo ""
echo "Baseline checkpoint: $BASELINE_CHECKPOINT"
echo "Quantization: 2-bit activations (LSQ)"
echo "Bin Regularization: Lambda = 2.0"
echo "Clip value: 6.0 (ReLU6 range)"
echo ""
echo "Expected BR effectiveness: 75-85% (with BatchNorm)"
echo "  - MNIST (no BN):  91.0%"
echo "  - MNIST (with BN): 85.5%"
echo "  - CIFAR-10: ???"
echo ""
echo "Training time: ~30-60 min (30 epochs)"
echo "======================================================================"
echo ""

python experiments/cifar10_mobilenet_qat_binreg.py \
    --pretrained "$BASELINE_CHECKPOINT" \
    --num-bits 2 \
    --clip-value 6.0 \
    --lambda-br 2.0 \
    --warmup-epochs 30 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.001 \
    --gpu 0 \
    --seed 42

# Note: Removed --manual-uniform-levels (defeats LSQ's data-driven learning)
# Now using proper S2 strategy: 30 epoch warmup, then BR with learned levels

echo ""
echo "======================================================================"
echo "QAT + BR training complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check TensorBoard for BR effectiveness metrics"
echo "  2. Compare activation histograms: baseline vs QAT+BR"
echo "  3. If effectiveness >80%: Excellent! BR works on real architectures"
echo "  4. If effectiveness 70-80%: Good! BN reduces but doesn't break BR"
echo "  5. If effectiveness <70%: May need to tune lambda or reduce BN"
echo "======================================================================"

