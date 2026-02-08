#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=100:00:00
#SBATCH --job-name=mnist_relu1_s43_missing

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

echo "=== Running MISSING seed 43 runs (8 runs) ==="
echo "1-bit: λ=1.0, λ=10.0"
echo "2-bit: λ=0.1, λ=1.0, λ=10.0" 
echo "4-bit: λ=0.1, λ=1.0, λ=10.0"

# Run 1-bit λ=1.0 and λ=10.0
"$ENV_PY" -u ./A-BR/experiments/mnist_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir ./A-BR/results/mnist_relu1_sweep/checkpoints \
    --seeds 43 \
    --bit-widths 1 \
    --lambdas 1.0 10.0 \
    --qat-epochs 30 \
    --warmup-epochs 10 \
    --batch-size 256 \
    --lr 0.002 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir ./A-BR/results/mnist_relu1_br_sweep

# Run all 2-bit
"$ENV_PY" -u ./A-BR/experiments/mnist_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir ./A-BR/results/mnist_relu1_sweep/checkpoints \
    --seeds 43 \
    --bit-widths 2 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 30 \
    --warmup-epochs 10 \
    --batch-size 256 \
    --lr 0.002 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir ./A-BR/results/mnist_relu1_br_sweep

# Run all 4-bit
"$ENV_PY" -u ./A-BR/experiments/mnist_automated_qat_br_sweep.py \
    --baseline-checkpoints-dir ./A-BR/results/mnist_relu1_sweep/checkpoints \
    --seeds 43 \
    --bit-widths 4 \
    --lambdas 0.1 1.0 10.0 \
    --qat-epochs 30 \
    --warmup-epochs 10 \
    --batch-size 128 \
    --lr 0.001 \
    --freeze-alpha \
    --br-backprop-to-alpha \
    --output-dir ./A-BR/results/mnist_relu1_br_sweep

