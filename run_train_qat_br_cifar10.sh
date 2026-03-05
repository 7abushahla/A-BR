#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=500:00:00
#SBATCH --job-name=qatbr_cifar10
#SBATCH --output=logs/slurm_qatbr_cifar10_%j.out


# Initialize conda
source ~/.bashrc

# Activate environment
conda activate /shared/b00090279/myenv

# Lock in the interpreter
ENV_PY="$CONDA_PREFIX/bin/python"

# Verify activation
if [[ "${CONDA_DEFAULT_ENV:-}" == "/shared/b00090279/myenv" || "${CONDA_DEFAULT_ENV:-}" == "b00090279" ]]; then
    echo "[YES] Conda environment activated: $CONDA_DEFAULT_ENV"
else
    echo "[NO] Conda environment did NOT activate."
    echo "Current python: $(which python || true)"
    exit 1
fi

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "Using Python: $ENV_PY"

# Sanity check
"$ENV_PY" -c "import sys, torch, os; \
print('sys.executable:', sys.executable); \
print('CONDA_PREFIX:', os.environ.get('CONDA_PREFIX')); \
print('torch:', torch.__version__); \
print('cuda available:', torch.cuda.is_available())"

# Run QAT+BR training
"$ENV_PY" -u ./experiments/train_qat_br_cifar10.py \
  --fp32_ckpt  ./checkpoints/cifar10_resnet18_baseline_20260105_213005.pth \
  --num_bits   2 \
  --lambda_br  0.1 \
  --warmup_epochs 10 \
  --qat_epochs    50