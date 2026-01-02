"""
LSQ (Learned Step-size Quantization) for Activations

Implements symmetric uniform quantization with learnable scale parameters
for activation quantization-aware training (QAT).

Based on: "Learned Step Size Quantization" (Esser et al., ICLR 2020)
Adapted from: LSQuantization-master/lsq.py (official implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============ Helper Functions (from official LSQ implementation) ============

def grad_scale(x, scale):
    """
    Scale gradient by a factor.
    
    Forward: pass through unchanged
    Backward: multiply gradient by scale
    """
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    """
    Straight-Through Estimator (STE) for rounding.
    
    Forward: round(x)
    Backward: pass gradient through unchanged (as if no rounding)
    """
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


# ============ LSQ Activation Quantizer ============

class LSQ_ActivationQuantizer(nn.Module):
    """
    Learned Step-size Quantization (LSQ) for unsigned activations [0, clip_value].
    
    Based on official LSQ implementation (ActLSQ) with adaptations for:
    - Unsigned activations (ReLU outputs)
    - Configurable clipping range
    - Data-driven initialization
    
    Args:
        num_bits: Number of bits for quantization (e.g., 2 for 4 levels)
        clip_value: Maximum activation value (e.g., 1.0 for ReLU1)
    """
    
    def __init__(self, num_bits=2, clip_value=1.0):
        super().__init__()
        self.num_bits = num_bits
        self.clip_value = clip_value
        
        # Unsigned quantization: Qn=0, Qp=2^nbits - 1
        self.Qn = 0
        self.Qp = 2 ** num_bits - 1
        
        # Learnable scale parameter (alpha in LSQ paper)
        # Initialize to a reasonable default based on expected activation range
        # For ReLU without clipping, assume range [0, ~4] initially
        # alpha = expected_max / Qp
        init_alpha = 4.0 / self.Qp if clip_value is None or clip_value > 10 else clip_value / self.Qp
        self.alpha = nn.Parameter(torch.tensor([init_alpha]))
        
        # Initialize flag (will be set on first forward pass)
        self.register_buffer('init_state', torch.zeros(1))
        
    def forward(self, x):
        """
        Quantize activations with learnable scale.
        
        Forward: x_q = round(x / alpha).clamp(Qn, Qp) * alpha
        Backward: STE for rounding, scaled gradient for alpha
        """
        # Data-driven initialization (on first batch)
        if self.training and self.init_state == 0:
            # Initialize alpha based on data statistics
            # alpha = 2 * mean(|x|) / sqrt(Qp)
            init_alpha = 2 * x.abs().mean() / math.sqrt(self.Qp)
            self.alpha.data.copy_(init_alpha)
            self.init_state.fill_(1)
            print(f"  [LSQ Init] num_bits={self.num_bits}, Qp={self.Qp}, x.mean={x.abs().mean().item():.4f}, alpha={init_alpha.item():.6f}")
        
        # Gradient scale factor for alpha (LSQ paper)
        # g = 1 / sqrt(numel * Qp)
        g = 1.0 / math.sqrt(x.numel() * self.Qp)
        
        # Scale alpha gradient
        alpha = grad_scale(self.alpha, g)
        
        # Quantize with STE (exactly as in original LSQ)
        # x_q = round(x / alpha).clamp(Qn, Qp) * alpha
        x_q = round_pass((x / alpha).clamp(self.Qn, self.Qp)) * alpha
        
        return x_q
    
    def get_quantization_levels(self):
        """Get current quantization levels."""
        levels = torch.arange(self.Qn, self.Qp + 1, dtype=self.alpha.dtype, device=self.alpha.device)
        return levels * self.alpha.data
    
    def extra_repr(self):
        return f'num_bits={self.num_bits}, Qp={self.Qp}, alpha={self.alpha.item():.4f}'


class QuantizedClippedReLU(nn.Module):
    """
    Clipped ReLU with LSQ quantization.
    
    Combines:
    1. ReLU activation
    2. Clipping to [0, clip_value]
    3. LSQ quantization with learnable scale
    
    This is used for QAT with Bin Regularization.
    
    IMPORTANT: For BR to work correctly, it needs access to PRE-quantization
    activations (continuous values after ReLU/clip, before round_pass).
    We store these in self.pre_quant_activation for BR loss computation.
    """
    
    def __init__(self, clip_value=1.0, num_bits=2):
        super().__init__()
        self.clip_value = clip_value
        self.num_bits = num_bits
        
        # LSQ quantizer
        self.quantizer = LSQ_ActivationQuantizer(
            num_bits=num_bits,
            clip_value=clip_value
        )
        
        # Store pre-quantization activations for BR
        self.pre_quant_activation = None
    
    def forward(self, x):
        # Step 1: ReLU + clip to [0, clip_value] - CONTINUOUS VALUES
        x_continuous = torch.clamp(F.relu(x), max=self.clip_value)
        
        # Store pre-quantization activations (for BR loss)
        self.pre_quant_activation = x_continuous
        
        # Step 2: Quantize with learnable scale - DISCRETE VALUES
        x_quantized = self.quantizer(x_continuous)
        
        return x_quantized
    
    def extra_repr(self):
        alpha = self.quantizer.alpha.item() if hasattr(self.quantizer, 'alpha') else 0
        return f'clip_value={self.clip_value}, num_bits={self.num_bits}, alpha={alpha:.4f}'

