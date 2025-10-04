import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.quantized_snn import QuantizedSNN

model = QuantizedSNN()

# Check layer 1 weights
with torch.no_grad():
    w1 = model.fc1.weight
    w1_quant = model.fc1.quantize_weight(w1)
    
    print("Layer 1 original weights:")
    print(f"  Min: {w1.min():.3f}, Max: {w1.max():.3f}")
    print(f"  Mean: {w1.mean():.3f}, Std: {w1.std():.3f}")
    
    print("\nLayer 1 quantized weights:")
    print(f"  Min: {w1_quant.min():.3f}, Max: {w1_quant.max():.3f}")
    print(f"  Mean: {w1_quant.mean():.3f}, Std: {w1_quant.std():.3f}")
    
    unique = torch.unique(w1_quant)
    print(f"  Unique values: {unique.tolist()}")
    
    # Count distribution
    for v in unique:
        count = (w1_quant == v).sum().item()
        pct = 100 * count / w1_quant.numel()
        print(f"    {v:.1f}V: {count} weights ({pct:.1f}%)")

# Test gradient flow
x = torch.randn(2, 784, requires_grad=True)
out = model(x)
loss = out.sum()
loss.backward()

print("\nGradient check:")
print(f"  fc1.weight.grad exists: {model.fc1.weight.grad is not None}")
if model.fc1.weight.grad is not None:
    print(f"  fc1.weight.grad mean: {model.fc1.weight.grad.mean():.6f}")
    print(f"  fc1.weight.grad std: {model.fc1.weight.grad.std():.6f}")