import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.baseline_snn import BaselineSNN
import json

def quantize_weights(weights, voltage_levels=[3.5, 4.0, 4.5, 5.0, 5.5]):
    """Quantize weights to discrete voltage levels"""
    levels = torch.tensor(voltage_levels, dtype=weights.dtype, device=weights.device)
    
    # Normalize to [0, 1]
    w_min = weights.min()
    w_max = weights.max()
    w_norm = (weights - w_min) / (w_max - w_min + 1e-8)
    
    # Scale to voltage range
    v_min = levels.min()
    v_max = levels.max()
    w_scaled = w_norm * (v_max - v_min) + v_min
    
    # Find nearest level
    w_expanded = w_scaled.unsqueeze(-1)
    distances = torch.abs(w_expanded - levels)
    indices = torch.argmin(distances, dim=-1)
    w_quantized = levels[indices]
    
    return w_quantized

def evaluate(model, test_loader, device='cpu'):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            img = img.view(img.size(0), -1)
            
            T = 10
            outputs = []
            for t in range(T):
                spike = (img > torch.rand_like(img)).float()
                out = model(spike)
                outputs.append(out)
            
            output = torch.stack(outputs).mean(0)
            pred = output.argmax(dim=1)
            
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            functional.reset_net(model)
    
    return 100. * correct / total

def main():
    device = 'cpu'
    
    # Load pretrained baseline
    print("Loading pretrained baseline model...")
    baseline = BaselineSNN().to(device)
    baseline.load_state_dict(torch.load('../results/models/baseline_snn.pth'))
    
    # Test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('../data/', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Baseline accuracy
    print("\nEvaluating baseline model...")
    baseline_acc = evaluate(baseline, test_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Quantize weights
    print("\nQuantizing weights to 5 voltage levels...")
    quantized = BaselineSNN().to(device)
    quantized.load_state_dict(baseline.state_dict())  # Copy weights
    
    with torch.no_grad():
        # Quantize fc1
        quantized.fc1.weight.data = quantize_weights(quantized.fc1.weight.data)
        
        # Quantize fc2
        quantized.fc2.weight.data = quantize_weights(quantized.fc2.weight.data)
    
    # Check quantization
    print("\nQuantized weight statistics:")
    print("Layer 1 (fc1):")
    unique1 = torch.unique(quantized.fc1.weight.data)
    print(f"  Unique values: {unique1.tolist()}")
    for v in unique1:
        count = (quantized.fc1.weight.data == v).sum().item()
        pct = 100 * count / quantized.fc1.weight.data.numel()
        print(f"    {v:.1f}V: {pct:.1f}%")
    
    print("Layer 2 (fc2):")
    unique2 = torch.unique(quantized.fc2.weight.data)
    print(f"  Unique values: {unique2.tolist()}")
    
    # Test quantized model
    print("\nEvaluating quantized model...")
    quantized_acc = evaluate(quantized, test_loader, device)
    print(f"Quantized accuracy: {quantized_acc:.2f}%")
    
    # Results
    accuracy_drop = baseline_acc - quantized_acc
    print(f"\nAccuracy drop: {accuracy_drop:.2f}%")
    print(f"Relative drop: {100*accuracy_drop/baseline_acc:.1f}%")
    
    # Save results
    results = {
        'baseline_accuracy': float(baseline_acc),
        'quantized_accuracy': float(quantized_acc),
        'accuracy_drop': float(accuracy_drop),
        'voltage_levels': [3.5, 4.0, 4.5, 5.0, 5.5],
        'method': 'post_training_quantization'
    }
    
    os.makedirs('../results/metrics', exist_ok=True)
    with open('../results/metrics/quantization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save quantized model
    torch.save(quantized.state_dict(), '../results/models/quantized_snn.pth')
    
    print("\n✓ Results saved!")

if __name__ == '__main__':
    main()