import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional

class BaselineSNN(nn.Module):
    """Simple 2-layer SNN without hardware constraints"""
    
    def __init__(self, input_size=784, hidden_size=300, output_size=10):
        super().__init__()
        
        # Layers
        self.fc1 = layer.Linear(input_size, hidden_size, bias=False)
        self.lif1 = neuron.LIFNode(tau=2.0)
        
        self.fc2 = layer.Linear(hidden_size, output_size, bias=False)
        self.lif2 = neuron.LIFNode(tau=2.0)
    
    def forward(self, x):
        # x shape: [Batch, 784]
        x = self.fc1(x)
        x = self.lif1(x)
        
        x = self.fc2(x)
        x = self.lif2(x)
        
        return x

# Test
if __name__ == '__main__':
    model = BaselineSNN()
    x = torch.randn(32, 784)  # batch=32, flattened 28x28
    
    # Run through time
    outputs = []
    for t in range(10):  # 10 timesteps
        out = model(x)
        outputs.append(out)
    
    functional.reset_net(model)  # Reset neuron states
    
    print("Model created successfully!")
    print(f"Output shape: {out.shape}")  # Should be [32, 10]