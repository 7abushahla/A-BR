"""
Sanity test: Verify BR gradient flow to alpha in coupled vs decoupled mode.

Expected behavior:
- Detached mode: alpha.grad should be None or 0
- Coupled mode: alpha.grad should be nonzero (from MSE-to-center term)
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abr.lsq_quantizer import QuantizedClippedReLU
from abr.regularizer_binreg import BinRegularizer
from abr.hooks import ActivationHookManager


def test_br_gradient_flow():
    """Test that BR loss can backprop to alpha in coupled mode."""
    
    print("="*70)
    print("BR Gradient Flow Sanity Test")
    print("="*70)
    
    # Create a simple model with one quantized ReLU
    model = nn.Sequential(
        nn.Linear(10, 10),
        QuantizedClippedReLU(clip_value=1.0, num_bits=2)
    )
    
    # Get the quantized ReLU module
    qrelu = model[1]
    alpha_param = qrelu.quantizer.alpha
    
    # Initialize alpha (mark as initialized to skip data-driven init)
    qrelu.quantizer.init_state.fill_(1)
    alpha_param.data.fill_(0.5)  # Set to known value
    
    print(f"\nInitial alpha: {alpha_param.item():.6f}")
    
    # Create dummy input
    batch_size = 32
    x = torch.randn(batch_size, 10)
    
    # Create hook manager and regularizer
    hook_manager = ActivationHookManager(
        model=model,
        target_modules=[QuantizedClippedReLU],
        layer_names=['1'],  # The QuantizedClippedReLU is at index 1
        detach_activations=False
    )
    
    regularizer = BinRegularizer(num_bits=2)
    
    print("\n" + "="*70)
    print("TEST 1: DETACHED MODE (alpha.item())")
    print("="*70)
    
    # Forward pass
    model.train()
    output = model(x)
    
    # Get pre-quant activations
    pre_quant_acts = hook_manager.get_pre_quant_activations()
    
    # Get alpha as Python float (DETACHED)
    alphas_detached = {'1': alpha_param.item()}
    
    # Compute BR loss (with L_CE = 0 for this test)
    br_loss_detached, _ = regularizer.compute_total_loss(pre_quant_acts, alphas_detached)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward (only BR loss, no task loss)
    br_loss_detached.backward()
    
    # Check alpha gradient
    alpha_grad_detached = alpha_param.grad
    
    print(f"BR loss: {br_loss_detached.item():.6f}")
    if alpha_grad_detached is None:
        print(f"Alpha gradient: None ✅ (expected for detached mode)")
    else:
        print(f"Alpha gradient: {alpha_grad_detached.item():.8f}")
        if abs(alpha_grad_detached.item()) < 1e-10:
            print("  → Effectively zero ✅ (expected for detached mode)")
        else:
            print("  → ⚠️  Nonzero! This is unexpected for detached mode.")
    
    print("\n" + "="*70)
    print("TEST 2: COUPLED MODE (alpha as tensor)")
    print("="*70)
    
    # Reset model state
    model.zero_grad()
    alpha_param.data.fill_(0.5)
    
    # Forward pass (fresh activations)
    output = model(x)
    pre_quant_acts = hook_manager.get_pre_quant_activations()
    
    # Get alpha as TENSOR (COUPLED)
    alphas_coupled = {'1': alpha_param.squeeze()}  # Keep as tensor
    
    # Compute BR loss
    br_loss_coupled, info = regularizer.compute_total_loss(pre_quant_acts, alphas_coupled)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward (only BR loss, no task loss)
    br_loss_coupled.backward()
    
    # Check alpha gradient
    alpha_grad_coupled = alpha_param.grad
    
    print(f"BR loss: {br_loss_coupled.item():.6f}")
    print(f"  - MSE component: {info['avg_mse']:.6f}")
    print(f"  - Var component: {info['avg_var']:.6f}")
    
    if alpha_grad_coupled is None:
        print(f"Alpha gradient: None ❌ (expected nonzero for coupled mode!)")
    else:
        print(f"Alpha gradient: {alpha_grad_coupled.item():.8f}")
        if abs(alpha_grad_coupled.item()) > 1e-10:
            print("  → Nonzero ✅ (expected for coupled mode)")
            print("  → BR can backprop into alpha/s")
        else:
            print("  → ⚠️  Effectively zero! BR is not backpropping to alpha.")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("Gradient to alpha comes ONLY from MSE-to-center term:")
    print("  mse_loss = (mean(bin) - bin_center)²")
    print("  where bin_center = levels[bin_idx] = idx * alpha")
    print("")
    print("Variance term does NOT contribute gradient to alpha:")
    print("  var_loss = var(bin_values)")
    print("  bin_values selected via stop-grad argmin")
    print("")
    
    if alpha_grad_coupled is not None and abs(alpha_grad_coupled.item()) > 1e-10:
        print("✅ PASS: Coupled mode shows gradient flow to alpha")
        print("         (Even though variance term doesn't contribute)")
    else:
        print("❌ FAIL: Coupled mode should have nonzero gradient to alpha")
    
    if alpha_grad_detached is None or abs(alpha_grad_detached.item()) < 1e-10:
        print("✅ PASS: Detached mode blocks gradient flow to alpha")
    else:
        print("⚠️  WARNING: Detached mode has nonzero gradient (unexpected)")
    
    print("="*70)


if __name__ == '__main__':
    test_br_gradient_flow()

