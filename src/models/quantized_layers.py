import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class QuantizedLinear(nn.Module):
    """Linear layer with weights quantized to device voltage levels"""
    
    def __init__(self, in_features, out_features, 
                 voltage_levels=[3.5, 4.0, 4.5, 5.0, 5.5]):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.voltage_levels = torch.tensor(voltage_levels, dtype=torch.float32)
        
        # Learnable continuous weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
    
    def quantize_weight(self, w):
        """Quantize continuous weights to discrete voltage levels"""
        # Normalize to [0, 1]
        w_min = w.min()
        w_max = w.max()
        w_norm = (w - w_min) / (w_max - w_min + 1e-8)
        
        # Scale to voltage range
        v_min = self.voltage_levels.min()
        v_max = self.voltage_levels.max()
        w_scaled = w_norm * (v_max - v_min) + v_min
        
        # Find nearest voltage level
        w_expanded = w_scaled.unsqueeze(-1)  # [..., 1]
        levels_expanded = self.voltage_levels.view(1, 1, -1)  # [1, 1, num_levels]
        
        distances = torch.abs(w_expanded - levels_expanded)
        indices = torch.argmin(distances, dim=-1)
        w_quantized = self.voltage_levels[indices]
        
        # Straight-through estimator: forward quantized, backward continuous
        if self.training:
        # Gradient flows through w, not w_quantized
            return w_scaled + (w_quantized - w_scaled).detach()
        else:
            return w_quantized
    
    def forward(self, x):
        w_quant = self.quantize_weight(self.weight)
        return F.linear(x, w_quant)

# Test
if __name__ == '__main__':
    # Test quantization
    layer = QuantizedLinear(784, 300)
    
    print("Original weight stats:")
    print(f"  Min: {layer.weight.min():.3f}")
    print(f"  Max: {layer.weight.max():.3f}")
    print(f"  Mean: {layer.weight.mean():.3f}")
    
    # Forward pass
    x = torch.randn(32, 784)
    out = layer(x)
    
    print(f"\nOutput shape: {out.shape}")
    
    # Check quantized values
    with torch.no_grad():
        w_quant = layer.quantize_weight(layer.weight)
        unique_vals = torch.unique(w_quant)
        print(f"\nQuantized unique values: {unique_vals.tolist()}")
        print(f"Expected: [3.5, 4.0, 4.5, 5.0, 5.5]")