#!/bin/bash
# Submit all 9 QAT+BR jobs to SLURM

echo "Submitting 9 QAT+BR jobs..."
echo ""

# ReLU
echo "=== ReLU ==="
sbatch relu_br_42.sh
sbatch relu_br_43.sh
sbatch relu_br_44.sh
echo ""

# ReLU1
echo "=== ReLU1 ==="
sbatch relu1_br_42.sh
sbatch relu1_br_43.sh
sbatch relu1_br_44.sh
echo ""

# ReLU6
echo "=== ReLU6 ==="
sbatch relu6_br_42.sh
sbatch relu6_br_43.sh
sbatch relu6_br_44.sh
echo ""

echo "✓ All 9 jobs submitted!"
echo ""
echo "Check status with: squeue -u $USER"

