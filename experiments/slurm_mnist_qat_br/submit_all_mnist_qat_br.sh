#!/bin/bash
# Submit all 3 MNIST QAT+BR jobs (one per activation type, each runs all 3 seeds)

echo "Submitting MNIST QAT+BR jobs..."

echo "=== ReLU (all seeds) ==="
sbatch relu_br_42.sh

echo "=== ReLU1 (all seeds) ==="
sbatch relu1_br_42.sh

echo "=== ReLU6 (all seeds) ==="
sbatch relu6_br_42.sh

echo ""
echo "All 3 jobs submitted!"
echo "Each job runs seeds 42, 43, 44 sequentially."
echo "Check status with: squeue -u \$USER"

