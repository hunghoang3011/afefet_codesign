import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
from .quantized_layers import QuantizedLinear

class QuantizedSNN(nn.Module):
    """SNN with quantized weights (5 voltage levels)"""
    
    def __init__(self, input_size=784, hidden_size=300, output_size=10):
        super().__init__()
        
        # Replace Linear with QuantizedLinear
        self.fc1 = QuantizedLinear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(tau=2.0, 
                                   surrogate_function=surrogate.ATan()) 
        
        self.fc2 = QuantizedLinear(hidden_size, output_size)
        self.lif2 = neuron.LIFNode(tau=2.0, 
                                   surrogate_function=surrogate.ATan()) 
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        
        x = self.fc2(x)
        x = self.lif2(x)
        
        return x

# Test
if __name__ == '__main__':
    model = QuantizedSNN()
    x = torch.randn(32, 784)
    
    outputs = []
    for t in range(10):
        out = model(x)
        outputs.append(out)
    
    functional.reset_net(model)
    
    print("Quantized SNN created successfully!")
    print(f"Output shape: {out.shape}")