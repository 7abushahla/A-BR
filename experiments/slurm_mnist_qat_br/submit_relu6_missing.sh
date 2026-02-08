#!/bin/bash
# Submit the 3 missing MNIST ReLU6 QAT+BR jobs

echo "Submitting MNIST ReLU6 QAT+BR missing jobs..."

echo "=== Seed 42 (2 missing runs: 4-bit λ=1.0, λ=10.0) ==="
sbatch relu6_br_42_missing.sh

echo "=== Seed 43 (all 9 runs) ==="
sbatch relu6_br_43.sh

echo "=== Seed 44 (all 9 runs) ==="
sbatch relu6_br_44.sh

echo ""
echo "All 3 jobs submitted!"
echo "Total missing runs: 2 (seed 42) + 9 (seed 43) + 9 (seed 44) = 20 runs"
echo "Check status with: squeue -u \$USER"

