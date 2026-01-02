"""
A-BR: Activation Bin Regularization

Implementation of Bin Regularization (BR) for activation quantization,
based on "Improving Low-Precision Network Quantization via Bin Regularization" (ICCV 2021).

Key components:
- regularizer_binreg: Bin Regularization loss
- lsq_quantizer: Learned Step-size Quantization (LSQ) for activations
- hooks: Activation hook manager for capturing pre/post-quantization values
"""

from .regularizer_binreg import BinRegularizer
from .lsq_quantizer import LSQ_ActivationQuantizer, QuantizedClippedReLU
from .hooks import ActivationHookManager

__all__ = [
    'BinRegularizer',
    'LSQ_ActivationQuantizer',
    'QuantizedClippedReLU',
    'ActivationHookManager',
]

