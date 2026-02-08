# SLURM Scripts for QAT+BR Sweep

This directory contains 9 SLURM batch scripts for running QAT+BR experiments in parallel.

## 📦 Files

### SLURM Scripts (9 total):
```
relu_br_42.sh      # ReLU, Seed 42
relu_br_43.sh      # ReLU, Seed 43
relu_br_44.sh      # ReLU, Seed 44

relu1_br_42.sh     # ReLU1, Seed 42
relu1_br_43.sh     # ReLU1, Seed 43
relu1_br_44.sh     # ReLU1, Seed 44

relu6_br_42.sh     # ReLU6, Seed 42
relu6_br_43.sh     # ReLU6, Seed 43
relu6_br_44.sh     # ReLU6, Seed 44
```

### Helper Script:
```
submit_all_qat_br.sh         # Submit all 9 jobs at once
```

---

## 🚀 Quick Start

### Option 1: Submit All at Once
```bash
cd experiments/slurm_qat_br
chmod +x submit_all_qat_br.sh
./submit_all_qat_br.sh
```

### Option 2: Submit Individually
```bash
cd experiments/slurm_qat_br

# Submit ReLU jobs
sbatch relu_br_42.sh
sbatch relu_br_43.sh
sbatch relu_br_44.sh

# Submit ReLU1 jobs
sbatch relu1_br_42.sh
sbatch relu1_br_43.sh
sbatch relu1_br_44.sh

# Submit ReLU6 jobs
sbatch relu6_br_42.sh
sbatch relu6_br_43.sh
sbatch relu6_br_44.sh
```

---

## 📊 What Each Job Does

Each job runs **9 QAT+BR trainings** (3 bit widths × 3 lambda values):

| Bit Width | Lambda | Training |
|-----------|--------|----------|
| 1         | 0.1    | 50 epochs (10 warmup + 40 BR) |
| 1         | 1.0    | 50 epochs |
| 1         | 10.0   | 50 epochs |
| 2         | 0.1    | 50 epochs |
| 2         | 1.0    | 50 epochs |
| 2         | 10.0   | 50 epochs |
| 4         | 0.1    | 50 epochs |
| 4         | 1.0    | 50 epochs |
| 4         | 10.0   | 50 epochs |

**Per job:** ~2.5 hours

---

## 📁 Output Locations

```
A-BR/results/
├── relu_br_sweep_seed42/
│   ├── checkpoints/         # 9 QAT models
│   └── results/
│       └── raw_results.json
├── relu_br_sweep_seed43/
├── relu_br_sweep_seed44/
├── relu1_br_sweep_seed42/
├── relu1_br_sweep_seed43/
├── relu1_br_sweep_seed44/
├── relu6_br_sweep_seed42/
├── relu6_br_sweep_seed43/
└── relu6_br_sweep_seed44/
```

---

## 🔍 Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# View output logs (while running or after completion)
tail -f slurm-<JOB_ID>.out

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

---

## 📈 After All Jobs Complete

### Step 1: Merge Results

```bash
cd A-BR

# Merge ReLU
python experiments/merge_qat_br_results.py \
    --results-dirs results/relu_br_sweep_seed42/results \
                   results/relu_br_sweep_seed43/results \
                   results/relu_br_sweep_seed44/results \
    --output-dir results/relu_br_sweep/results

# Merge ReLU1
python experiments/merge_qat_br_results.py \
    --results-dirs results/relu1_br_sweep_seed42/results \
                   results/relu1_br_sweep_seed43/results \
                   results/relu1_br_sweep_seed44/results \
    --output-dir results/relu1_br_sweep/results

# Merge ReLU6
python experiments/merge_qat_br_results.py \
    --results-dirs results/relu6_br_sweep_seed42/results \
                   results/relu6_br_sweep_seed43/results \
                   results/relu6_br_sweep_seed44/results \
    --output-dir results/relu6_br_sweep/results
```

### Step 2: View Results

```bash
# View merged tables
cat results/relu_br_sweep/results/summary.txt
cat results/relu1_br_sweep/results/summary.txt
cat results/relu6_br_sweep/results/summary.txt
```

---

## ⚙️ Configuration

Each script uses:
- **Memory:** 16G (increased from 4G for QAT training)
- **Time:** 100 hours (max allowed)
- **GPUs:** 1 per job
- **Partition:** gpu

Adjust `#SBATCH` directives in each script if needed.

---

## 🎛️ Alpha Control Options (BR Package)

**Note:** For the new `BR/` package (not A-BR), we now have separate freeze controls:

### **Old (A-BR, still valid):**
```bash
--freeze-alpha  # Freezes all alphas (both W and A)
```

### **New (BR package):**
```bash
--freeze-weight-alpha  # Freeze weight alphas only
--freeze-act-alpha     # Freeze activation alphas only
```

### **Usage Examples:**

#### **Paper-Faithful W-BR (No Freeze):**
```bash
python BR/experiments/cifar10_qat_br.py \
    --num-bits 2 \
    --lambda-br 1.0 \
    --lambda-br-act 0.0 \
    --warmup-epochs 30 \
    --qat-epochs 100
    # No freeze = W alpha co-evolves (paper approach)
```

#### **W+A BR with A-freeze Only (Recommended):**
```bash
python BR/experiments/cifar10_qat_br.py \
    --num-bits 2 \
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-act-alpha
    # W alpha adapts, A alpha frozen (stable)
```

#### **Both Frozen (Maximum Stability):**
```bash
python BR/experiments/cifar10_qat_br.py \
    --num-bits 2 \
    --lambda-br 1.0 \
    --lambda-br-act 1.0 \
    --warmup-epochs 30 \
    --qat-epochs 100 \
    --freeze-weight-alpha \
    --freeze-act-alpha
    # Both fixed during BR phase
```

### **Decision Matrix:**

| `--freeze-weight-alpha` | `--freeze-act-alpha` | W alpha updates? | A alpha updates? | Use Case |
|-------------------------|----------------------|------------------|------------------|----------|
| ❌ No | ❌ No | CE + W-BR | CE only* | Recommended (W adapts, A stable) |
| ❌ No | ✅ Yes | CE + W-BR | ❌ None | W-BR adapts, A-BR very stable |
| ✅ Yes | ❌ No | ❌ None | CE only* | W-BR stable, A-BR adapts (unusual) |
| ✅ Yes | ✅ Yes | ❌ None | ❌ None | Maximum stability (both fixed) |

*Unless `--br-backprop-to-alpha-act` is set

### **Lambda Control (Toggle W-BR and A-BR):**

```bash
# W-only BR (original paper)
--lambda-br 1.0 --lambda-br-act 0.0

# W+A BR (full package)
--lambda-br 1.0 --lambda-br-act 1.0

# A-only BR (research)
--lambda-br 0.0 --lambda-br-act 1.0

# Different strengths
--lambda-br 1.0 --lambda-br-act 10.0  # Stronger A-BR
```

### **Key Differences:**
- **W-BR:** Always allows BR gradients to weight alpha (paper-faithful, weights are stable)
- **A-BR:** Gradients detached by default (activations are batch-dependent)
- **Independent control:** Each lambda and freeze flag works independently

---

## ⚠️ Prerequisites

Before submitting jobs, make sure:

1. ✅ PTQ sweep completed for all 3 activation types
2. ✅ Baseline checkpoints exist:
   - `A-BR/results/relu_sweep/checkpoints/*seed42*.pth`
   - `A-BR/results/relu_sweep/checkpoints/*seed43*.pth`
   - `A-BR/results/relu_sweep/checkpoints/*seed44*.pth`
   - (same for relu1 and relu6)

3. ✅ Conda environment `/shared/b00090279/myenv` is set up
4. ✅ All dependencies installed in conda env

---

## 🎯 Expected Runtime

- **Per job:** ~2.5 hours (9 trainings × 15 min each)
- **All 9 jobs in parallel:** ~2.5 hours total
- **Total GPU hours:** 9 jobs × 2.5 hours = 22.5 GPU hours

---

## 📧 Email Notifications (Optional)

To receive email notifications when jobs start/finish, add these lines to each `.sh` file:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com
```

---

**Good luck with your experiments! 🚀**

