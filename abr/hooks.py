"""
Activation Hook Manager for A-BR

Utilities for hooking into forward pass and capturing activations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable


class ActivationHookManager:
    """
    Manages forward hooks for capturing activations during training.
    
    This class helps register hooks on specific layers to capture their
    output activations for kurtosis regularization.
    
    Args:
        model: PyTorch model
        target_modules: List of module types to hook (e.g., [nn.ReLU6])
        layer_names: Optional list of specific layer names to hook
        exclude_first_last: Whether to exclude first and last occurrences
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[type]] = None,
        layer_names: Optional[List[str]] = None,
        exclude_first_last: bool = True,
        detach_activations: bool = False
    ):
        self.model = model
        self.target_modules = target_modules or [nn.ReLU6, nn.ReLU]
        self.layer_names = layer_names
        self.exclude_first_last = exclude_first_last
        self.detach_activations = detach_activations  # For gradient flow control
        
        # Storage
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.registered_layers: List[str] = []
        
        # Setup hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        # Find all target layers
        all_target_layers = []
        
        if self.layer_names:
            # Use explicitly specified layer names
            for name, module in self.model.named_modules():
                if name in self.layer_names:
                    all_target_layers.append((name, module))
        else:
            # Auto-detect based on module types
            for name, module in self.model.named_modules():
                if any(isinstance(module, target_type) for target_type in self.target_modules):
                    all_target_layers.append((name, module))
        
        # Optionally exclude first and last
        if self.exclude_first_last and len(all_target_layers) > 2:
            all_target_layers = all_target_layers[1:-1]
        
        # Register hooks
        for name, module in all_target_layers:
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
            self.registered_layers.append(name)
        
        print(f"A-BR: Registered hooks on {len(self.registered_layers)} layers:")
        for name in self.registered_layers:
            print(f"  - {name}")
    
    def _make_hook(self, name: str) -> Callable:
        """
        Create a forward hook function for a specific layer.
        
        Args:
            name: Layer name
            
        Returns:
            Hook function
        """
        def hook(module, input, output):
            # Store the output activations
            # Check detach flag dynamically (not at hook creation time)
            if self.detach_activations:
                self.activations[name] = output.detach().clone()
            else:
                # Keep output connected to graph for gradient flow
                self.activations[name] = output
        
        return hook
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get the currently stored activations.
        
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        return self.activations
    
    def get_pre_quant_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get PRE-quantization activations from QuantizedClippedReLU modules.
        
        This is critical for Bin Regularization (BR) to work correctly.
        BR must operate on continuous post-activation, pre-quantization values,
        NOT on the discrete post-quantization values.
        
        Returns:
            Dictionary mapping layer names to PRE-quantization activation tensors
        """
        from abr.lsq_quantizer import QuantizedClippedReLU
        
        pre_quant_acts = {}
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizedClippedReLU) and name in self.registered_layers:
                if hasattr(module, 'pre_quant_activation') and module.pre_quant_activation is not None:
                    pre_quant_acts[name] = module.pre_quant_activation
        
        return pre_quant_acts
    
    def clear_activations(self):
        """Clear stored activations to free memory."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.registered_layers.clear()
        print("A-BR: All hooks removed")
    
    def get_registered_layers(self) -> List[str]:
        """Get list of layers with registered hooks."""
        return self.registered_layers.copy()
    
    def set_training_mode(self, training: bool):
        """
        Set training mode for hook manager.
        
        Args:
            training: If True, keep gradients. If False, detach activations.
        """
        self.detach_activations = not training
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks."""
        self.remove_hooks()
        self.clear_activations()
    
    def __repr__(self):
        return (f"ActivationHookManager("
                f"registered_layers={len(self.registered_layers)}, "
                f"target_modules={[m.__name__ for m in self.target_modules]})")


class SelectiveActivationHookManager(ActivationHookManager):
    """
    Variant that allows more selective hooking with filtering criteria.
    """
    
    def __init__(
        self,
        model: nn.Module,
        filter_fn: Optional[Callable[[str, nn.Module], bool]] = None,
        **kwargs
    ):
        """
        Args:
            model: PyTorch model
            filter_fn: Function that takes (name, module) and returns True to include
            **kwargs: Additional arguments for base class
        """
        self.filter_fn = filter_fn
        super().__init__(model, **kwargs)
    
    def _register_hooks(self):
        """Register hooks with custom filtering."""
        all_target_layers = []
        
        for name, module in self.model.named_modules():
            # Check if module matches target types
            is_target_type = any(isinstance(module, target_type) 
                               for target_type in self.target_modules)
            
            # Apply custom filter if provided
            if self.filter_fn:
                passes_filter = self.filter_fn(name, module)
            else:
                passes_filter = True
            
            if is_target_type and passes_filter:
                all_target_layers.append((name, module))
        
        # Optionally exclude first and last
        if self.exclude_first_last and len(all_target_layers) > 2:
            all_target_layers = all_target_layers[1:-1]
        
        # Register hooks
        for name, module in all_target_layers:
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
            self.registered_layers.append(name)
        
        print(f"A-BR: Registered hooks on {len(self.registered_layers)} layers")

