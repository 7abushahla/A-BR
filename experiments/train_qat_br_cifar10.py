#!/usr/bin/env python3
"""
train_qat_br_cifar10.py
=======================
Standalone QAT + Bin-Regularization training script for CIFAR-10 ResNet18.

Saves a checkpoint that BR_SNN_CIFAR10.ipynb can load directly via
EXISTING_QAT_BR_CKPT.

Typical usage
-------------
# From the repo root with your conda env active:
python A-BR/experiments/train_qat_br_cifar10.py \
    --fp32_ckpt  A-BR/checkpoints/cifar10_resnet18_baseline_20260105_213005.pth \
    --num_bits   2 \
    --lambda_br  0.1 \
    --warmup_epochs 10 \
    --qat_epochs    50

# Higher BR regularisation to push closer to T=3 SNN accuracy:
python A-BR/experiments/train_qat_br_cifar10.py \
    --fp32_ckpt  A-BR/checkpoints/cifar10_resnet18_baseline_20260105_213005.pth \
    --lambda_br  2.0 \
    --warmup_epochs 10 \
    --qat_epochs    50

Checkpoint format
-----------------
{
  'epoch':                  int,
  'model_state_dict':       OrderedDict,
  'optimizer_state_dict':   OrderedDict,
  'test_acc':               float,
  'best_acc':               float,
  'num_bits':               int,
  'clip_value':             float,
  'lambda_br':              float,
  'warmup_epochs':          int,
  'freeze_alpha':           bool,
  'br_backprop_alpha':      bool,
  'br_effectiveness':       float,   # same formula as BinRegularizer (train set, last epoch)
  'br_quant_mse':           float,
  'fp32_baseline_ckpt':     str,
}
"""

import sys, os, argparse, random
from datetime import datetime
from pathlib import Path

# ── Resolve repo root so A-BR is importable wherever the script is called from ─
_SCRIPT_DIR = Path(__file__).resolve().parent        # A-BR/experiments/
_ABR_DIR    = _SCRIPT_DIR.parent                     # A-BR/
_THESIS_DIR = _ABR_DIR.parent                        # repo root

sys.path.insert(0, str(_ABR_DIR))

# ── Redirect torch.hub / torchvision weight downloads inside the repo ──────────
_CACHE_DIR = _THESIS_DIR / '.cache' / 'torch'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('TORCH_HOME', str(_CACHE_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights

import numpy as np
from tqdm import tqdm

from abr.lsq_quantizer import LSQ_ActivationQuantizer, QuantizedClippedReLU
from abr.regularizer_binreg import BinRegularizer
from abr.hooks import ActivationHookManager


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='QAT + Bin Regularization training for CIFAR-10 ResNet18',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument('--fp32_ckpt', required=True,
                   help='Path to a CIFAR-10 FP32 ResNet18 checkpoint to start from.')
    # Quantisation
    p.add_argument('--num_bits',   type=int,   default=2,   help='Activation bit width.')
    p.add_argument('--clip_value', type=float, default=6.0, help='QuantizedClippedReLU ceiling.')
    # BR
    p.add_argument('--lambda_br',         type=float, default=0.1,  help='BR loss weight.')
    p.add_argument('--no_freeze_alpha',   action='store_true',       help='Keep α trainable throughout.')
    p.add_argument('--no_br_backprop_alpha', action='store_true',    help='Detach α from BR gradient.')
    # Schedule
    p.add_argument('--warmup_epochs', type=int,   default=10,  help='Epochs of task-loss-only warmup.')
    p.add_argument('--qat_epochs',    type=int,   default=50,  help='Epochs of QAT+BR training.')
    p.add_argument('--batch_size',    type=int,   default=128)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    # I/O
    p.add_argument('--data_dir',  default=str(_ABR_DIR / 'data'),
                   help='Directory for CIFAR-10 dataset.')
    p.add_argument('--ckpt_dir',  default=str(_ABR_DIR / 'checkpoints'),
                   help='Directory to save checkpoints.')
    p.add_argument('--num_workers', type=int, default=4,
                   help='DataLoader workers (use 0 inside Jupyter).')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model helpers
# ══════════════════════════════════════════════════════════════════════════════

def _replace_activations(module, src_types, dst_factory):
    """Recursively replace all children that are instances of src_types."""
    for name, child in module.named_children():
        if isinstance(child, tuple(src_types)):
            setattr(module, name, dst_factory())
        else:
            _replace_activations(child, src_types, dst_factory)


def build_resnet18_cifar10_qat(num_bits, clip_value, fp32_baseline_ckpt):
    """
    ResNet18 adapted for CIFAR-10 with QuantizedClippedReLU activations,
    initialised from a FP32 CIFAR-10 checkpoint (non-quantizer params only).
    """
    model = resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)

    _replace_activations(
        model,
        src_types=[nn.ReLU, nn.ReLU6],
        dst_factory=lambda: QuantizedClippedReLU(clip_value=clip_value, num_bits=num_bits),
    )

    ckpt = torch.load(fp32_baseline_ckpt, map_location='cpu', weights_only=False)
    sd   = ckpt.get('model_state_dict', ckpt)
    msd  = model.state_dict()
    ok   = {k: v for k, v in sd.items()
            if k in msd and v.shape == msd[k].shape and 'quantizer' not in k}
    msd.update(ok)
    model.load_state_dict(msd, strict=False)
    print(f'  ✓ Loaded {len(ok)} tensors from FP32 baseline: {fp32_baseline_ckpt}')
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_ann(model, loader, device, desc='Eval'):
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total   += y.size(0)
    return correct / total


def train_one_epoch(model, loader, optimizer, criterion,
                    hook_manager, regularizer,
                    lambda_br, use_br, br_backprop_alpha, device):
    model.train()
    hook_manager.set_training_mode(True)
    tot_loss = tot_task = tot_br = 0.0
    correct = total = 0

    for x, y in tqdm(loader, desc='Train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        out       = model(x)
        task_loss = criterion(out, y)

        if use_br:
            acts   = hook_manager.get_pre_quant_activations()
            alphas = {n: (m.quantizer.alpha.squeeze()
                          if br_backprop_alpha
                          else m.quantizer.alpha.item())
                      for n, m in model.named_modules()
                      if isinstance(m, QuantizedClippedReLU)}
            br_loss, _ = regularizer.compute_total_loss(acts, alphas)
            loss       = task_loss + lambda_br * br_loss
            tot_br    += br_loss.item() if torch.is_tensor(br_loss) else br_loss
        else:
            loss = task_loss

        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        tot_task += task_loss.item()
        correct  += out.detach().argmax(1).eq(y).sum().item()
        total    += y.size(0)

    n = len(loader)
    return tot_loss/n, tot_task/n, tot_br/n, correct/total


@torch.no_grad()
def test_one_epoch(model, loader, criterion,
                   hook_manager, regularizer, lambda_br, use_br, device):
    model.eval()
    hook_manager.set_training_mode(False)
    tot_loss = tot_task = tot_br = 0.0
    correct = total = 0
    br_eff = br_mse = n_br = 0

    for x, y in tqdm(loader, desc='Test', leave=False):
        x, y = x.to(device), y.to(device)
        out       = model(x)
        task_loss = criterion(out, y)

        if use_br:
            acts   = hook_manager.get_pre_quant_activations()
            alphas = {n: m.quantizer.alpha.item()
                      for n, m in model.named_modules()
                      if isinstance(m, QuantizedClippedReLU)}
            br_loss, info = regularizer.compute_total_loss(acts, alphas)
            loss   = task_loss + lambda_br * br_loss
            tot_br += br_loss.item() if torch.is_tensor(br_loss) else br_loss
            br_eff += info.get('avg_effectiveness', 0)
            br_mse += info.get('avg_quantization_mse', 0)
            n_br   += 1
        else:
            loss = task_loss

        tot_loss += loss.item()
        tot_task += task_loss.item()
        correct  += out.argmax(1).eq(y).sum().item()
        total    += y.size(0)

    n = len(loader)
    br_info = {'eff': br_eff/n_br if n_br else 0.0,
               'mse': br_mse/n_br if n_br else 0.0}
    return tot_loss/n, tot_task/n, tot_br/n, correct/total, br_info


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Device & seeds ────────────────────────────────────────────────────────
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    FREEZE_ALPHA      = not args.no_freeze_alpha
    BR_BACKPROP_ALPHA = not args.no_br_backprop_alpha
    TOTAL_EPOCHS      = args.warmup_epochs + args.qat_epochs
    T_ALIGN           = 2 ** args.num_bits - 1

    print(f'device: {device}')
    print(f'torch {torch.__version__} | torchvision {torchvision.__version__}')
    print(f'T_ALIGN = {T_ALIGN}  (2^{args.num_bits}-1)  |  '
          f'λ_BR = {args.lambda_br}  |  '
          f'epochs = {TOTAL_EPOCHS} ({args.warmup_epochs}w + {args.qat_epochs}q)')
    print()

    # ── Data ──────────────────────────────────────────────────────────────────
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10(root=args.data_dir, train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == 'cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=(device == 'cuda'))
    print(f'CIFAR-10 | train {len(train_ds):,}  test {len(test_ds):,}\n')

    # ── Model ─────────────────────────────────────────────────────────────────
    qat_model = build_resnet18_cifar10_qat(
        num_bits=args.num_bits,
        clip_value=args.clip_value,
        fp32_baseline_ckpt=args.fp32_ckpt,
    ).to(device)

    all_qrelu = [n for n, m in qat_model.named_modules()
                 if isinstance(m, QuantizedClippedReLU)]

    hook_manager = ActivationHookManager(
        model=qat_model,
        target_modules=[QuantizedClippedReLU],
        layer_names=all_qrelu,
        exclude_first_last=False,
        detach_activations=False,
    )
    regularizer = BinRegularizer(num_bits=args.num_bits)
    optimizer   = torch.optim.Adam(qat_model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    criterion   = nn.CrossEntropyLoss()
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
                      optimizer, T_max=args.qat_epochs, eta_min=args.lr * 0.01)

    # ── Output checkpoint path ────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_path = ckpt_dir / f'cifar10_resnet18_qat_br_b{args.num_bits}_lbr{args.lambda_br}_{timestamp}.pth'
    print(f'Checkpoint → {ckpt_path}\n')

    best_acc     = 0.0
    alpha_frozen = False

    print(f'QAT model: {sum(p.numel() for p in qat_model.parameters()):,} params')
    print(f'Hooks: {len(hook_manager.registered_layers)} layers\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(TOTAL_EPOCHS):
        is_warmup = epoch < args.warmup_epochs
        use_br    = not is_warmup
        stage     = 'WARMUP' if is_warmup else 'QAT+BR'

        # Freeze α at the warmup / QAT+BR boundary
        if FREEZE_ALPHA and epoch == args.warmup_epochs and not alpha_frozen:
            alpha_frozen = True
            print('\n──── Freezing LSQ α after warmup ────')
            for n, m in qat_model.named_modules():
                if isinstance(m, QuantizedClippedReLU):
                    m.quantizer.alpha.requires_grad_(False)
                    print(f'  {n}: α = {m.quantizer.alpha.item():.5f}  [FROZEN]')
            print('─────────────────────────────────────\n')

        tr_loss, tr_task, tr_br, tr_acc = train_one_epoch(
            qat_model, train_loader, optimizer, criterion,
            hook_manager, regularizer, args.lambda_br, use_br, BR_BACKPROP_ALPHA, device)

        te_loss, te_task, te_br, te_acc, br_info = test_one_epoch(
            qat_model, test_loader, criterion,
            hook_manager, regularizer, args.lambda_br, use_br, device)

        if epoch >= args.warmup_epochs:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        br_str = (f'  br_eff={br_info["eff"]:.1f}%  q-mse={br_info["mse"]:.5f}'
                  if use_br else '')
        print(f'[{epoch+1:3d}/{TOTAL_EPOCHS}] {stage:7s}  lr={lr:.5f}  '
              f'tr_task={tr_task:.4f} tr_acc={tr_acc:.4f}  '
              f'te_task={te_task:.4f} te_acc={te_acc:.4f}' + br_str)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     qat_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc':             te_acc,
                'best_acc':             best_acc,
                'num_bits':             args.num_bits,
                'clip_value':           args.clip_value,
                'lambda_br':            args.lambda_br,
                'warmup_epochs':        args.warmup_epochs,
                'freeze_alpha':         FREEZE_ALPHA,
                'br_backprop_alpha':    BR_BACKPROP_ALPHA,
                'br_effectiveness':     br_info['eff'],
                'br_quant_mse':         br_info['mse'],
                'fp32_baseline_ckpt':   args.fp32_ckpt,
            }, ckpt_path)
            print(f'  ✓ Best: {best_acc:.4f}  → {ckpt_path}')

    hook_manager.remove_hooks()
    print(f'\nTraining done.  Best test acc: {best_acc:.4f}')
    print(f'Checkpoint saved to: {ckpt_path}')
    print(f'\nTo load in the notebook set:')
    print(f'  EXISTING_QAT_BR_CKPT = "{ckpt_path}"')


if __name__ == '__main__':
    main()
