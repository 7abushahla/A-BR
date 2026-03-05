#!/bin/bash
#
# Complete Experiment: Baseline PTQ vs QAT-BR
#
# This script runs the full comparison:
# 1. Train baseline FP32 model
# 2. Train QAT-BR model
# 3. Evaluate both with quantization
# 4. Generate comparison plots and metrics
#

set -e  # Exit on error

NUM_BITS=2
EPOCHS_BASELINE=30
EPOCHS_QAT=30
WARMUP_EPOCHS=10
LAMBDA_BR=100.0  # Paper-faithful (sum, not average)
SEED=42

echo "========================================================================"
echo "COMPLETE EXPERIMENT: Baseline PTQ vs QAT-BR"
echo "========================================================================"
echo "Configuration:"
echo "  - Quantization: ${NUM_BITS}-bit"
echo "  - Baseline epochs: ${EPOCHS_BASELINE}"
echo "  - QAT epochs: ${EPOCHS_QAT} (warmup: ${WARMUP_EPOCHS})"
echo "  - Lambda BR: ${LAMBDA_BR}"
echo "  - Seed: ${SEED}"
echo "========================================================================"
echo ""

# ===== Step 1: Train Baseline FP32 Model =====
echo "[Step 1/3] Training Baseline FP32 Model..."
echo "----------------------------------------"
python experiments/mnist_baseline.py \
    --epochs ${EPOCHS_BASELINE} \
    --lr 0.001 \
    --clip-value 1.0 \
    --seed ${SEED}

echo ""
echo "✓ Baseline training complete"
echo ""

# ===== Step 2: Train QAT-BR Model =====
echo "[Step 2/3] Training QAT-BR Model..."
echo "----------------------------------------"
python experiments/mnist_qat_binreg.py \
    --num-bits ${NUM_BITS} \
    --warmup-epochs ${WARMUP_EPOCHS} \
    --epochs ${EPOCHS_QAT} \
    --lambda-br ${LAMBDA_BR} \
    --freeze-alpha \
    --seed ${SEED}

echo ""
echo "✓ QAT-BR training complete"
echo ""

# ===== Step 3: Evaluate Both Models =====
echo "[Step 3/3] Evaluating Quantization: Baseline PTQ vs QAT-BR..."
echo "----------------------------------------"

# Find the most recent checkpoints
BASELINE_MODEL=$(ls -t checkpoints/mnist_baseline_*.pth | head -1)
QAT_MODEL=$(ls -t checkpoints/mnist_qat_binreg_*.pth | head -1)

echo "Using models:"
echo "  Baseline: ${BASELINE_MODEL}"
echo "  QAT-BR:   ${QAT_MODEL}"
echo ""

python experiments/evaluate_quantization.py \
    --baseline-model ${BASELINE_MODEL} \
    --qat-model ${QAT_MODEL} \
    --num-bits ${NUM_BITS} \
    --calibration-batches 50 \
    --batch-size 256 \
    --mode both \
    --output-dir results/quantization_comparison_${NUM_BITS}bit

echo ""
echo "========================================================================"
echo "EXPERIMENT COMPLETE!"
echo "========================================================================"
echo "Results saved to: results/quantization_comparison_${NUM_BITS}bit/"
echo ""
echo "View comparison plots:"
echo "  ls results/quantization_comparison_${NUM_BITS}bit/*.png"
echo ""
echo "View summary:"
echo "  cat results/quantization_comparison_${NUM_BITS}bit/summary.txt"
echo "========================================================================"

