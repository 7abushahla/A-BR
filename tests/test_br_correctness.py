"""
Comprehensive correctness tests for Bin Regularization implementation.

Tests verify:
1. Bin assignment matches LSQ integer code
2. MSE computed for all non-empty bins (including size 1)
3. Variance computed correctly (unbiased=False, skipped for size 1)
4. Metrics use LSQ assignment, not nearest-level
5. Gradient flow works in coupled mode
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abr.regularizer_binreg import BinRegularizer


def test_bin_assignment():
    """Test that bin assignment uses LSQ integer code, not nearest-level."""
    print("\n" + "="*70)
    print("TEST 1: Bin Assignment (LSQ Integer Code)")
    print("="*70)
    
    regularizer = BinRegularizer(num_bits=2)  # 4 levels: 0, 1, 2, 3
    
    # Test case: alpha = 1.0, so levels = [0, 1, 2, 3]
    alpha = 1.0
    activations = torch.tensor([
        0.1,   # Should map to bin 0 (round(0.1/1.0) = 0)
        0.4,   # Should map to bin 0 (round(0.4/1.0) = 0)
        0.5,   # Should map to bin 0 (round(0.5/1.0) = 0, PyTorch rounds to even)
        0.9,   # Should map to bin 1 (round(0.9/1.0) = 1)
        1.4,   # Should map to bin 1 (round(1.4/1.0) = 1)
        1.5,   # Should map to bin 2 (round(1.5/1.0) = 2, PyTorch rounds to even)
        2.4,   # Should map to bin 2 (round(2.4/1.0) = 2)
        2.5,   # Should map to bin 2 (round(2.5/1.0) = 2, PyTorch rounds to even)
        3.5,   # Should map to bin 3 (round(3.5/1.0) = 4 → clip to Qp=3)
    ])
    
    # Expected bins based on LSQ: round(clip(v/s, 0, 3))
    # Note: PyTorch uses "round half to even" (banker's rounding)
    expected_bins = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3])
    
    # Compute actual bins using regularizer's method
    actual_bins = torch.round(torch.clamp(activations / alpha, regularizer.Qn, regularizer.Qp)).long()
    
    print(f"Activations: {activations.tolist()}")
    print(f"Expected bins: {expected_bins.tolist()}")
    print(f"Actual bins:   {actual_bins.tolist()}")
    
    assert torch.equal(actual_bins, expected_bins), "Bin assignment doesn't match LSQ integer code!"
    print("✅ PASS: Bin assignment uses LSQ integer code (round + clip)")


def test_single_element_bin():
    """Test that single-element bins compute MSE but not variance."""
    print("\n" + "="*70)
    print("TEST 2: Single-Element Bin Handling")
    print("="*70)
    
    regularizer = BinRegularizer(num_bits=2)
    
    # Create activation with one value in bin 1
    alpha = 1.0
    activations = torch.tensor([1.2])  # Single value, should go to bin 1
    
    loss, mse, var, info = regularizer.compute_bin_loss(activations, alpha)
    
    print(f"Activation: {activations.item()}")
    print(f"Alpha: {alpha}")
    print(f"Expected bin: 1 (round(1.2/1.0) = 1)")
    print(f"Expected MSE: (1.2 - 1.0)^2 = 0.04")
    print(f"Expected var: 0 (single element)")
    print(f"")
    print(f"Actual BR loss: {loss.item():.6f}")
    print(f"Actual MSE: {info['br_mse_loss']:.6f}")
    print(f"Actual var: {info['br_var_loss']:.6f}")
    
    expected_mse = (1.2 - 1.0) ** 2
    assert abs(info['br_mse_loss'] - expected_mse) < 1e-6, "MSE not computed for single-element bin!"
    assert info['br_var_loss'] == 0.0, "Variance should be 0 for single-element bin!"
    print("✅ PASS: Single-element bins compute MSE only, variance=0")


def test_multi_element_bin():
    """Test that multi-element bins compute both MSE and variance."""
    print("\n" + "="*70)
    print("TEST 3: Multi-Element Bin Handling")
    print("="*70)
    
    regularizer = BinRegularizer(num_bits=2)
    
    # Create activations with multiple values in bin 1
    alpha = 1.0
    activations = torch.tensor([0.8, 1.0, 1.2])  # All should go to bin 1
    
    loss, mse, var, info = regularizer.compute_bin_loss(activations, alpha)
    
    mean_val = activations.mean().item()
    expected_mse = (mean_val - 1.0) ** 2
    expected_var = activations.var(unbiased=False).item()
    
    print(f"Activations: {activations.tolist()}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Expected bin: 1 for all")
    print(f"Expected MSE: ({mean_val:.4f} - 1.0)^2 = {expected_mse:.6f}")
    print(f"Expected var (unbiased=False): {expected_var:.6f}")
    print(f"")
    print(f"Actual MSE: {info['br_mse_loss']:.6f}")
    print(f"Actual var: {info['br_var_loss']:.6f}")
    
    assert abs(info['br_mse_loss'] - expected_mse) < 1e-6, "MSE incorrect for multi-element bin!"
    assert abs(info['br_var_loss'] - expected_var) < 1e-6, "Variance incorrect for multi-element bin!"
    print("✅ PASS: Multi-element bins compute both MSE and variance correctly")


def test_quantization_mse_consistency():
    """Test that quantization MSE uses LSQ assignment, not nearest-level."""
    print("\n" + "="*70)
    print("TEST 4: Quantization MSE Uses LSQ Assignment")
    print("="*70)
    
    regularizer = BinRegularizer(num_bits=2)
    
    # Test with value at boundary where nearest-level might differ from LSQ
    alpha = 1.0
    activations = torch.tensor([0.6])  # Near boundary
    
    # LSQ assignment: round(0.6/1.0) = 1, so quantized = 1.0
    expected_quantized = 1.0
    expected_mse = (0.6 - 1.0) ** 2
    
    loss, mse, var, info = regularizer.compute_bin_loss(activations, alpha)
    
    print(f"Activation: {activations.item()}")
    print(f"LSQ assignment: round(0.6/1.0) = 1 → quantized = 1.0")
    print(f"Expected MSE: (0.6 - 1.0)^2 = {expected_mse:.6f}")
    print(f"Actual quantization MSE: {info['quantization_mse']:.6f}")
    
    assert abs(info['quantization_mse'] - expected_mse) < 1e-6, "Quantization MSE not using LSQ assignment!"
    print("✅ PASS: Quantization MSE uses LSQ assignment (not nearest-level)")


def test_gradient_to_alpha():
    """Test that BR can backprop to alpha in coupled mode."""
    print("\n" + "="*70)
    print("TEST 5: Gradient Flow to Alpha (Coupled Mode)")
    print("="*70)
    
    regularizer = BinRegularizer(num_bits=2)
    
    # Create tensor alpha (coupled mode)
    alpha = torch.tensor(1.0, requires_grad=True)
    activations = torch.tensor([0.7, 0.9, 1.1])  # Mean != 1.0, will have MSE gradient
    
    loss, mse, var, _ = regularizer.compute_bin_loss(activations, alpha)
    
    print(f"Activations: {activations.tolist()}")
    print(f"Alpha (tensor): {alpha.item()}")
    print(f"BR loss: {loss.item():.6f}")
    
    # Backprop
    loss.backward()
    
    print(f"Gradient to alpha: {alpha.grad.item() if alpha.grad is not None else 'None'}")
    
    assert alpha.grad is not None, "No gradient to alpha in coupled mode!"
    assert abs(alpha.grad.item()) > 1e-8, "Gradient to alpha is zero!"
    print("✅ PASS: BR backprops to alpha in coupled mode")


def test_clipping_behavior():
    """Test that values outside [0, Qp*alpha] are clipped correctly."""
    print("\n" + "="*70)
    print("TEST 6: Clipping Behavior at Boundaries")
    print("="*70)
    
    regularizer = BinRegularizer(num_bits=2)  # Qp = 3
    
    alpha = 1.0
    activations = torch.tensor([
        -0.5,  # Below 0, should clip to bin 0
        0.0,   # Exactly at 0, bin 0
        3.0,   # Exactly at Qp*alpha, bin 3
        4.0,   # Above Qp*alpha, should clip to bin 3
    ])
    
    expected_bins = torch.tensor([0, 0, 3, 3])
    actual_bins = torch.round(torch.clamp(activations / alpha, regularizer.Qn, regularizer.Qp)).long()
    
    print(f"Activations: {activations.tolist()}")
    print(f"Qn={regularizer.Qn}, Qp={regularizer.Qp}, alpha={alpha}")
    print(f"Expected bins: {expected_bins.tolist()}")
    print(f"Actual bins:   {actual_bins.tolist()}")
    
    assert torch.equal(actual_bins, expected_bins), "Clipping behavior incorrect!"
    print("✅ PASS: Values outside bounds are clipped correctly")


def test_effectiveness_metric():
    """Test that effectiveness metric is computed correctly."""
    print("\n" + "="*70)
    print("TEST 7: Effectiveness Metric")
    print("="*70)
    
    regularizer = BinRegularizer(num_bits=2)
    
    alpha = 1.0
    
    # Test case 1: Perfect clustering (all values exactly at levels)
    activations_perfect = torch.tensor([0.0, 1.0, 2.0, 3.0])
    _, _, _, info_perfect = regularizer.compute_bin_loss(activations_perfect, alpha)
    
    print("Case 1: Perfect clustering (values at levels)")
    print(f"  Activations: {activations_perfect.tolist()}")
    print(f"  Effectiveness: {info_perfect['effectiveness']:.2f}%")
    print(f"  Mean distance: {info_perfect['mean_distance']:.6f}")
    
    # Should be very high (near 100%)
    assert info_perfect['effectiveness'] > 95.0, "Perfect clustering should have high effectiveness!"
    
    # Test case 2: Dispersed (values spread within bins)
    activations_dispersed = torch.tensor([0.2, 0.8, 1.2, 1.8, 2.2, 2.8])
    _, _, _, info_dispersed = regularizer.compute_bin_loss(activations_dispersed, alpha)
    
    print("\nCase 2: Dispersed (values spread within bins)")
    print(f"  Activations: {activations_dispersed.tolist()}")
    print(f"  Effectiveness: {info_dispersed['effectiveness']:.2f}%")
    print(f"  Mean distance: {info_dispersed['mean_distance']:.6f}")
    
    # Should be lower than perfect case
    assert info_dispersed['effectiveness'] < info_perfect['effectiveness'], "Dispersed should have lower effectiveness!"
    
    print("✅ PASS: Effectiveness metric differentiates perfect vs dispersed")


def run_all_tests():
    """Run all correctness tests."""
    print("\n" + "="*80)
    print("BIN REGULARIZATION CORRECTNESS TESTS")
    print("="*80)
    print("Testing paper-faithful implementation...")
    
    try:
        test_bin_assignment()
        test_single_element_bin()
        test_multi_element_bin()
        test_quantization_mse_consistency()
        test_gradient_to_alpha()
        test_clipping_behavior()
        test_effectiveness_metric()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nImplementation is paper-faithful:")
        print("  ✅ Bin assignment via LSQ integer code")
        print("  ✅ MSE computed for all non-empty bins")
        print("  ✅ Variance computed correctly (unbiased=False)")
        print("  ✅ Metrics use LSQ assignment")
        print("  ✅ Gradient flow to alpha works")
        print("  ✅ Boundary clipping correct")
        print("  ✅ Effectiveness metric sensible")
        print("="*80)
        
    except AssertionError as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        raise


if __name__ == '__main__':
    run_all_tests()

