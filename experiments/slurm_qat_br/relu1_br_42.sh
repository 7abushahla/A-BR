#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --job-name=relu1_br_s42

# Initialize conda
source ~/.bashrc

# Activate environment
conda activate /shared/b00090279/myenv

# Lock in the interpreter
ENV_PY="$CONDA_PREFIX/bin/python"

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "/shared/b00090279/myenv" || "$CONDA_DEFAULT_ENV" == "b00090279" ]]; then
    echo "[YES] Conda environment activated: $CONDA_DEFAULT_ENV"
else
    echo "[NO] Conda environment did NOT activate."
    echo "Current python: $(which python)"
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
"$ENV_PY" -u ./A-BR/experiments/cifar10_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir ./A-BR/results/relu1_sweep/checkpoints \
    --seeds 42 \
    --bit-widths 1 2 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 50 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir ./A-BR/results/relu1_br_sweep_seed42

