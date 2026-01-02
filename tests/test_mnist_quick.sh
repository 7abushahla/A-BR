#!/bin/bash

# Quick MNIST training test to verify end-to-end functionality
# Runs for just a few epochs to ensure everything works

set -e  # Exit on error

echo "========================================="
echo "QUICK MNIST END-TO-END TEST"
echo "========================================="
echo ""

cd "$(dirname "$0")/.."

echo "Test 1: Coupled mode (BR backprops to alpha) - 2 epochs"
echo "---------------------------------------------------------"
python experiments/mnist_qat_binreg.py \
    --num-bits 4 \
    --warmup-epochs 1 \
    --epochs 2 \
    --lambda-br 0.1 \
    --br-backprop-to-alpha \
    --seed 42

echo ""
echo "✅ Test 1 PASSED: Coupled mode training completed"
echo ""

echo "Test 2: Decoupled mode (BR doesn't affect alpha) - 2 epochs"
echo "-------------------------------------------------------------"
python experiments/mnist_qat_binreg.py \
    --num-bits 4 \
    --warmup-epochs 1 \
    --epochs 2 \
    --lambda-br 0.1 \
    --seed 42

echo ""
echo "✅ Test 2 PASSED: Decoupled mode training completed"
echo ""

echo "========================================="
echo "ALL END-TO-END TESTS PASSED! ✅"
echo "========================================="
echo ""
echo "Summary:"
echo "  ✅ Coupled mode (paper-faithful) works"
echo "  ✅ Decoupled mode (original) works"
echo "  ✅ No runtime errors"
echo "  ✅ Ready for full training experiments"

