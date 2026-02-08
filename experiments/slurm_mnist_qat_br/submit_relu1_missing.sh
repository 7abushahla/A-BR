#!/bin/bash
# Submit the 2 missing MNIST ReLU1 QAT+BR jobs

echo "Submitting MNIST ReLU1 QAT+BR missing jobs..."

echo "=== Seed 43 (8 missing runs) ==="
echo "    1-bit: λ=1.0, λ=10.0"
echo "    2-bit: all lambdas"
echo "    4-bit: all lambdas"
sbatch relu1_br_43_missing.sh

echo "=== Seed 44 (all 9 runs) ==="
sbatch relu1_br_44.sh

echo ""
echo "All 2 jobs submitted!"
echo "Total missing runs: 8 (seed 43) + 9 (seed 44) = 17 runs"
echo "Check status with: squeue -u \$USER"

