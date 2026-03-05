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
    Clipped ReLU with LSQ quantization + optional Rate-Code Noise Injection (RCNI).

    Combines:
    1. ReLU activation
    2. Clipping to [0, clip_value]
    3. LSQ quantization with learnable scale
    4. (Optional) Rate-Code Noise Injection for T-step SNN robustness

    This is used for QAT with Bin Regularization.

    IMPORTANT: For BR to work correctly, it needs access to PRE-quantization
    activations (continuous values after ReLU/clip, before round_pass).
    We store these in self.pre_quant_activation for BR loss computation.

    Rate-Code Noise Injection (RCNI)
    ---------------------------------
    When `t_align` is set (e.g. t_align=3 for 2-bit), the quantised output has
    Uniform(−α/(2·T), +α/(2·T)) noise added during training.  This noise is
    analytically derived from the worst-case T-step rate-coding quantisation
    error: an activation within ±α/(2T) of a bin boundary can be assigned the
    wrong spike count, producing a ±α error in the reconstructed activation.

    By training every downstream layer with this noise present, the model learns
    to be robust to the inter-layer cascade errors that degrade SNN accuracy at
    small T — eliminating the need for a separate SNN fine-tuning phase.

    RCNI does NOT change BR loss computation (BR still sees the clean
    pre_quant_activation) and does NOT affect evaluation (noise is training-only).
    """

    def __init__(self, clip_value=1.0, num_bits=2, t_align=None):
        """
        Args:
            clip_value: Upper bound for the ReLU clip (default 1.0).
            num_bits:   Activation bit width (default 2).
            t_align:    Simulation timesteps for RCNI noise (None = disabled).
                        Set to 2**num_bits - 1 for T-aligned conversion.
        """
        super().__init__()
        self.clip_value = clip_value
        self.num_bits   = num_bits
        self.t_align    = t_align   # None → RCNI disabled; int → RCNI enabled

        # LSQ quantizer
        self.quantizer = LSQ_ActivationQuantizer(
            num_bits=num_bits,
            clip_value=clip_value
        )

        # Store pre-quantization activations for BR
        self.pre_quant_activation = None

    def forward(self, x):
        # ── Step 1: ReLU + clip to [0, clip_value] – continuous values ────────
        if self.clip_value is not None:
            x_continuous = torch.clamp(F.relu(x), max=self.clip_value)
        else:
            x_continuous = F.relu(x)

        # Store pre-quantization activations (for BR loss, unaffected by noise)
        self.pre_quant_activation = x_continuous

        # ── Step 2: LSQ hard quantisation → {0, α, 2α, …, Qp·α} ─────────────
        x_quantized = self.quantizer(x_continuous)

        # ── Step 3: Rate-Code Noise Injection (RCNI) – training only ─────────
        # Noise magnitude = α / (2·T):  the maximum rate-coding error for a
        # T-step IFNode is ±α (one bin off), which occurs only for activations
        # within ±1/(2T) of a bin boundary.  Injecting Uniform(−α/(2T), +α/(2T))
        # at every layer trains downstream weights to absorb these cascade errors.
        if self.training and self.t_align is not None and self.t_align > 0:
            alpha    = self.quantizer.alpha.detach()
            Qp       = self.quantizer.Qp
            sigma    = alpha / (2.0 * self.t_align)              # α / (2T)
            noise    = (torch.rand_like(x_quantized) - 0.5) * 2.0 * sigma
            # Clamp to valid quantisation range so noise can't push outside [0, Qp·α]
            x_quantized = torch.clamp(x_quantized + noise,
                                       torch.zeros_like(alpha),
                                       alpha * Qp)

        return x_quantized

    def extra_repr(self):
        alpha = self.quantizer.alpha.item() if hasattr(self.quantizer, 'alpha') else 0
        rcni  = f', RCNI T={self.t_align}' if self.t_align is not None else ''
        return f'clip_value={self.clip_value}, num_bits={self.num_bits}, alpha={alpha:.4f}{rcni}'
    
    def extra_repr(self):
        alpha = self.quantizer.alpha.item() if hasattr(self.quantizer, 'alpha') else 0
        return f'clip_value={self.clip_value}, num_bits={self.num_bits}, alpha={alpha:.4f}'

