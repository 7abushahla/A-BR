"""
Bin Regularization for Activation Quantization

Encourages activations to cluster tightly around quantization bin centers,
creating sharp (Dirac-like) distributions for each bin.

Based on: "Improving Low-Precision Network Quantization via Bin Regularization" (ICCV 2021)

Key Insight from BR Paper:
- LSQ learns WHERE the quantization grid should be (via learned step size s)
- BR makes activations STICK to that grid (minimize within-bin variance)
- They are NOT redundant: LSQ defines optimal grid, BR shapes distribution

Paper-Faithful Implementation:
- BR uses the SAME levels as LSQ: [0, s, 2s, ..., Qp·s]
- Bin assignment: round(clip(v/s, Qn, Qp)) - LSQ's integer code (Eq. 1-2)
- Loss per bin: L_mse(mean, target) + L_var (Eq. 5)
- Single-element bins: MSE only, variance=0 (V_i ≤ 1)
- Multi-element bins: Both MSE and variance (var with unbiased=False)
- Empty bins: Skipped
- Levels are DYNAMIC (based on LSQ's current learned s)

Differences from Paper:
- Paper targets WEIGHTS; we apply to ACTIVATIONS (post-ReLU, unsigned)
- Paper sums over bins; we AVERAGE across layers (makes λ scale-invariant)
- Effective λ is scaled by 1/num_layers compared to paper's formulation
- Unsigned quantization only [0, Qp] (paper also discusses signed)

Two-Stage Training (BR Paper S2 Strategy):
1. Warmup (~30 epochs): LSQ learns optimal s from data (no BR)
2. Joint training: Add BR while continuing to optimize s (co-evolution)
"""

import torch
import torch.nn as nn


class BinRegularizer(nn.Module):
    """
    Bin Regularization (BR) for activations.
    
    Implements the BR paper's per-bin loss formulation:
    L_BR = Σ (L_mse(⟨v_i⟩, v̂_i) + L_var(v_i))
           i=1 to 2^b
    
    Where:
    - v̂_i = i·s (LSQ's quantization levels, NOT fixed linspace!)
    - Bins determined by LSQ integer code: round(clip(v/s, n, p))
    - L_mse: Push bin mean toward that bin's LSQ target
    - L_var: { 0             if V_i ≤ 1
             { var(v_i)      if V_i ≥ 2  (unbiased=False)
    
    Implementation notes:
    - BR uses the SAME levels as LSQ (dynamic, based on learned s/alpha)
    - We average loss across layers (paper sums over bins within a layer)
    - This makes λ scale-invariant to network depth
    - Effective λ is scaled by 1/num_layers vs paper
    - Applied to post-ReLU activations (unsigned quantization)
    
    The lambda weighting is applied in the training loop: L = L_CE + λ · L_BR
    
    Args:
        num_bits: Target bit-width for quantization (e.g., 2 for 4 levels)
    """
    
    def __init__(self, num_bits=2):
        super().__init__()
        self.num_bits = num_bits
        
        # UNSIGNED quantization bounds (for post-ReLU activations)
        # IMPORTANT: This assumes activations are in [0, ∞) (clipped ReLU)
        # For signed activations (pre-ReLU, residuals), would need Qn = -(2^(b-1))
        self.Qn = 0
        self.Qp = 2 ** num_bits - 1
        self.num_levels = 2 ** num_bits
        
        print(f"BinRegularizer: {num_bits}-bit ({self.num_levels} levels)")
        print(f"  Levels are DYNAMIC: [0, α, 2α, ..., {self.Qp}α] (tied to LSQ)")
        print(f"  ⚠️  Unsigned quantization only (for post-ReLU activations)")
    
    def compute_bin_loss(self, activations: torch.Tensor, alpha: float) -> tuple:
        """
        Compute bin regularization loss for a single activation tensor.
        
        CRITICAL SEMANTIC CLARITY:
        - 'alpha' here is the STEP SIZE (s in BR paper notation)
        - NOT the clip value or activation range!
        - LSQ quantization: x_q = round(x / s) * s
        - Quantization levels: [0, s, 2s, 3s, ..., Qp·s]
        - BR targets are these SAME levels (using LSQ's learned s)
        
        This function fetches the CURRENT alpha/s from LSQ each forward pass,
        ensuring BR and LSQ always use the same grid (no semantic mismatch).
        
        Args:
            activations: Activation tensor (any shape)
            alpha: Current learned step size (s) from LSQ quantizer for this layer
            
        Returns:
            (total_loss, mse_loss, var_loss, bin_info)
        """
        # Flatten activations
        acts_flat = activations.flatten()
        
        # Compute quantization levels dynamically from LSQ's current alpha
        # levels = [0, α, 2α, 3α, ..., Qp·α]
        level_indices = torch.arange(self.num_levels, device=acts_flat.device, dtype=acts_flat.dtype)
        levels = level_indices * alpha
        # GRADIENT FLOW NOTE:
        # - If alpha is a tensor (not .item()), gradients can flow: loss → levels → alpha
        # - This allows BR to contribute ∂L_BR/∂s term (paper-faithful "simultaneous update")
        # - Bin assignment (argmin below) is non-differentiable, so bin membership is stop-grad
        # - This prevents chaotic gradients while still letting BR guide grid position
        
        # For each activation, find bin assignment using LSQ integer code (paper-faithful)
        # Paper: bin_i = round(clip(v/s, Qn, Qp))
        # This is both faster (no distance matrix for assignment) and more faithful to paper
        bin_assignments = torch.round(torch.clamp(acts_flat / alpha, self.Qn, self.Qp)).long()
        # NOTE: round() is non-differentiable (returns integer indices)
        # This means bin membership is automatically stop-grad (stable)
        # Only the target levels (bin_center = bin_idx * alpha) can receive gradients
        
        # Compute loss for each bin
        total_mse = 0.0
        total_var = 0.0
        bins_used = 0
        
        for bin_idx in range(self.num_levels):
            # Get activations assigned to this bin
            mask = (bin_assignments == bin_idx)
            bin_values = acts_flat[mask]
            
            if len(bin_values) == 0:
                # Empty bin, skip
                continue
            
            bins_used += 1
            bin_center = levels[bin_idx]
            
            # L_mse: Push bin mean toward target (paper Eq. 5)
            # Use .mean() consistently for all non-empty bins (even size 1)
            mse_loss = ((bin_values.mean() - bin_center) ** 2)
            total_mse += mse_loss
            
            # L_var: Minimize variance (make sharp)
            # Paper: variance term becomes 0 when V_i ≤ 1
            if len(bin_values) >= 2:
                # Use unbiased=False explicitly (paper just says "var")
                # At 2-bit with small bins, unbiased estimator can be noisy
                var_loss = bin_values.var(unbiased=False)
                total_var += var_loss
            # else: var_loss = 0 (implicit, V_i = 1)
        
        # Total BR loss for this layer: Σ(L_mse + L_var) across bins
        # Paper: L_BR = Σ(L_mse(⟨v_i⟩, v̂_i) + L_var(v_i))
        loss = total_mse + total_var
        
        # ========== Compute BR Effectiveness Metrics ==========
        # Metrics computed without gradients (logging only)
        with torch.no_grad():
            # Use ACTUAL LSQ quantization (same as bin assignment) for metrics
            # This ensures metrics reflect true quantization error, not nearest-level
            acts_quantized = bin_assignments.float() * alpha
            
            # 1. Mean distance to quantized value (actual LSQ error)
            mean_distance = (acts_flat - acts_quantized).abs().mean()
            
            # 2. BR Effectiveness Score: 0-100%
            # Perfect clustering (Dirac deltas) = 100%
            # Uniform spread = 0%
            # Tensor-safe: ensure max_dist is tensor with correct device/dtype
            max_dist = (alpha * 0.5) if torch.is_tensor(alpha) else torch.tensor(
                alpha * 0.5, device=acts_flat.device, dtype=acts_flat.dtype
            )
            effectiveness = 100.0 * (1.0 - mean_distance / (max_dist + 1e-12))
            effectiveness = effectiveness.clamp(0.0, 100.0)
            
            # 3. Percentage of activations "at" quantization levels (within 1% of alpha)
            threshold = (0.01 * alpha) if torch.is_tensor(alpha) else torch.tensor(
                0.01 * alpha, device=acts_flat.device, dtype=acts_flat.dtype
            )
            near_levels = ((acts_flat - acts_quantized).abs() < threshold).float().mean() * 100.0
            
            # 4. Actual Quantization MSE (using LSQ assignment, not nearest-level)
            quantization_mse = ((acts_flat - acts_quantized) ** 2).mean()
        
        # Info for logging
        bin_info = {
            'bins_used': bins_used,
            'br_mse_loss': total_mse.item() if isinstance(total_mse, torch.Tensor) else total_mse,  # BR loss component
            'br_var_loss': total_var.item() if isinstance(total_var, torch.Tensor) else total_var,  # BR loss component
            'quantization_mse': quantization_mse.item(),  # ACTUAL quantization error!
            'mean_distance': mean_distance.item(),
            'effectiveness': effectiveness.item(),
            'pct_near_levels': near_levels.item(),
        }
        
        return loss, total_mse, total_var, bin_info
    
    def compute_total_loss(self, activations_dict: dict, alphas_dict: dict) -> tuple:
        """
        Compute bin regularization loss across all layers.
        
        Args:
            activations_dict: Dictionary of {layer_name: activation_tensor}
            alphas_dict: Dictionary of {layer_name: current_alpha_value}
                         These are the learned step sizes from LSQ quantizers
            
        Returns:
            (total_loss, info_dict)
        """
        total_loss = 0.0
        total_mse = 0.0
        total_var = 0.0
        total_quant_mse = 0.0  # NEW: Actual quantization MSE
        total_mean_distance = 0.0
        total_effectiveness = 0.0
        total_pct_near = 0.0
        layer_losses = {}
        
        for layer_name, acts in activations_dict.items():
            # Get the current alpha for this layer from LSQ
            if layer_name not in alphas_dict:
                continue  # Skip if alpha not provided
            alpha = alphas_dict[layer_name]
            
            loss, mse, var, bin_info = self.compute_bin_loss(acts, alpha)
            
            total_loss += loss
            total_mse += mse if isinstance(mse, torch.Tensor) else torch.tensor(mse)
            total_var += var if isinstance(var, torch.Tensor) else torch.tensor(var)
            total_quant_mse += bin_info['quantization_mse']  # NEW: Accumulate actual MSE
            total_mean_distance += bin_info['mean_distance']
            total_effectiveness += bin_info['effectiveness']
            total_pct_near += bin_info['pct_near_levels']
            
            layer_losses[layer_name] = {
                'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
                'mse': mse.item() if isinstance(mse, torch.Tensor) else mse,
                'var': var.item() if isinstance(var, torch.Tensor) else var,
                'quantization_mse': bin_info['quantization_mse'],  # NEW: Actual quantization error
                'bins_used': bin_info['bins_used'],
                'mean_distance': bin_info['mean_distance'],
                'effectiveness': bin_info['effectiveness'],
                'pct_near_levels': bin_info['pct_near_levels'],
            }
        
        num_layers = len(activations_dict)
        # NOTE: We average across layers rather than summing (as paper does over bins)
        # This means our λ is effectively scaled by 1/num_layers compared to paper
        # Rationale: Makes λ scale-invariant to number of layers
        avg_loss = total_loss / num_layers if num_layers > 0 else total_loss
        avg_mse = total_mse / num_layers if num_layers > 0 else total_mse
        avg_var = total_var / num_layers if num_layers > 0 else total_var
        avg_quant_mse = total_quant_mse / num_layers if num_layers > 0 else 0.0  # NEW: Average actual MSE
        avg_mean_distance = total_mean_distance / num_layers if num_layers > 0 else 0.0
        avg_effectiveness = total_effectiveness / num_layers if num_layers > 0 else 0.0
        avg_pct_near = total_pct_near / num_layers if num_layers > 0 else 0.0
        
        info_dict = {
            'avg_loss': avg_loss.item() if isinstance(avg_loss, torch.Tensor) else avg_loss,
            'avg_mse': avg_mse.item() if isinstance(avg_mse, torch.Tensor) else avg_mse,
            'avg_var': avg_var.item() if isinstance(avg_var, torch.Tensor) else avg_var,
            'avg_quantization_mse': avg_quant_mse,  # NEW: Actual quantization MSE!
            'avg_mean_distance': avg_mean_distance,
            'avg_effectiveness': avg_effectiveness,
            'avg_pct_near': avg_pct_near,
            'layer_losses': layer_losses
        }
        
        return avg_loss, info_dict
    
    def get_bin_statistics(self, activations_dict: dict, alphas_dict: dict) -> dict:
        """
        Get detailed statistics about bin assignments (for visualization/debugging).
        
        Args:
            activations_dict: Dictionary of {layer_name: activation_tensor}
            alphas_dict: Dictionary of {layer_name: current_alpha_value}
        
        Returns dictionary with per-layer bin counts and statistics.
        """
        stats = {}
        
        for layer_name, acts in activations_dict.items():
            if layer_name not in alphas_dict:
                continue
            
            acts_flat = acts.flatten()
            alpha = alphas_dict[layer_name]
            
            # Compute levels dynamically
            level_indices = torch.arange(self.num_levels, device=acts_flat.device, dtype=acts_flat.dtype)
            levels = level_indices * alpha
            
            # Bin assignment (paper-faithful: LSQ integer code)
            bin_assignments = torch.round(torch.clamp(acts_flat / alpha, self.Qn, self.Qp)).long()
            
            # Count per bin
            bin_counts = []
            bin_means = []
            bin_stds = []
            
            for bin_idx in range(self.num_levels):
                mask = (bin_assignments == bin_idx)
                bin_values = acts_flat[mask]
                
                bin_counts.append(len(bin_values))
                if len(bin_values) > 0:
                    bin_means.append(bin_values.mean().item())
                    bin_stds.append(bin_values.std().item() if len(bin_values) > 1 else 0.0)
                else:
                    bin_means.append(0.0)
                    bin_stds.append(0.0)
            
            stats[layer_name] = {
                'bin_counts': bin_counts,
                'bin_means': bin_means,
                'bin_stds': bin_stds,
                'total_values': len(acts_flat)
            }
        
        return stats

