import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class BaselineSNN(nn.Module):
    def __init__(self, T=20):
        super().__init__()
        self.T = T
        
        self.fc1 = layer.Linear(784, 300, bias=False)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        
        self.fc2 = layer.Linear(300, 10, bias=False)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        return x

def quantize_weights_to_voltage(weights, voltage_levels=[3.5, 4.0, 4.5, 5.0, 5.5], base_voltage=4.5):
    """
    Hardware-aware quantization that maps weights to discrete AFeFET voltage levels.

    Key insight: weights are SCALING FACTORS for the base voltage, not absolute values.
    - Original weight w represents: effective_voltage = base_voltage * (1 + w * alpha)
    - We quantize to discrete voltage levels that the device can physically implement
    - alpha is optimized to minimize quantization error (MSE)

    Args:
        weights: Original continuous weights
        voltage_levels: Discrete voltage levels the AFeFET device can operate at
        base_voltage: Reference voltage (typically middle of range)
    """
    voltage_levels = torch.tensor(voltage_levels, dtype=weights.dtype, device=weights.device)

    # Step 1: Find optimal alpha by grid search to minimize quantization MSE
    w_min, w_max = weights.min(), weights.max()
    v_min, v_max = voltage_levels.min(), voltage_levels.max()

    # Candidate alphas based on weight range
    alpha_candidates = []
    for percentile in [90, 95, 99, 99.5, 99.9]:
        w_percentile = torch.quantile(torch.abs(weights), percentile / 100.0)
        if w_percentile > 0:
            alpha = (v_max - base_voltage) / (base_voltage * w_percentile)
            alpha_candidates.append(alpha)

    # Add alphas based on min/max
    if w_max > 0:
        alpha_candidates.append((v_max - base_voltage) / (base_voltage * w_max))
    if w_min < 0:
        alpha_candidates.append((base_voltage - v_min) / (base_voltage * abs(w_min)))

    # Test each alpha and pick the one with minimum MSE
    best_alpha = None
    best_mse = float('inf')

    for alpha in alpha_candidates:
        target_voltages = base_voltage * (1 + weights * alpha)
        v_expanded = target_voltages.unsqueeze(-1)
        distances = torch.abs(v_expanded - voltage_levels)
        indices = torch.argmin(distances, dim=-1)
        quantized_voltages = voltage_levels[indices]
        quantized_weights = (quantized_voltages / base_voltage - 1) / alpha

        mse = ((weights - quantized_weights) ** 2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    # Step 2: Use best alpha for final quantization
    alpha = best_alpha
    target_voltages = base_voltage * (1 + weights * alpha)

    # Step 3: Quantize to nearest available device voltage
    v_expanded = target_voltages.unsqueeze(-1)
    distances = torch.abs(v_expanded - voltage_levels)
    indices = torch.argmin(distances, dim=-1)
    quantized_voltages = voltage_levels[indices]

    # Step 4: Convert back to weight space
    quantized_weights = (quantized_voltages / base_voltage - 1) / alpha

    return quantized_weights, alpha

def evaluate(model, test_loader, device, T=20):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            
            img = img.view(img.size(0), -1)
            
            spike_img = []
            for t in range(T):
                spike = torch.rand_like(img) < img.clamp(0, 1)
                spike_img.append(spike.float())
            spike_img = torch.stack(spike_img)
            
            output = model(spike_img)
            output_mean = output.mean(0)
            pred = output_mean.argmax(dim=1)
            
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            functional.reset_net(model)
    
    return 100. * correct / total

def main():
    device = 'cpu'
    T = 20
    
    print("Loading pretrained baseline model...")
    baseline = BaselineSNN(T=T).to(device)
    baseline.load_state_dict(torch.load('../results/models/baseline_snn_v2.pth'))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('../data/', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("\nEvaluating baseline model...")
    baseline_acc = evaluate(baseline, test_loader, device, T)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    voltage_levels = [3.5, 4.0, 4.5, 5.0, 5.5]
    base_voltage = 4.5

    print(f"\nQuantizing weights to voltage levels: {voltage_levels}")
    print(f"Base voltage: {base_voltage}V")

    quantized = BaselineSNN(T=T).to(device)
    quantized.load_state_dict(baseline.state_dict())

    with torch.no_grad():
        quantized.fc1.weight.data, alpha1 = quantize_weights_to_voltage(
            quantized.fc1.weight.data, voltage_levels, base_voltage
        )
        quantized.fc2.weight.data, alpha2 = quantize_weights_to_voltage(
            quantized.fc2.weight.data, voltage_levels, base_voltage
        )

    print(f"\nScaling factors:")
    print(f"  Layer 1 (fc1): alpha = {alpha1:.6f}")
    print(f"  Layer 2 (fc2): alpha = {alpha2:.6f}")
    
    print("\nQuantized weight statistics (with voltage mapping):")
    print("Layer 1 (fc1):")
    unique1 = torch.unique(quantized.fc1.weight.data).sort()[0]
    print(f"  Weight values → Voltages:")
    for v in unique1:
        voltage = base_voltage * (1 + v * alpha1)
        count = (quantized.fc1.weight.data == v).sum().item()
        pct = 100 * count / quantized.fc1.weight.data.numel()
        print(f"    w={v:.4f} → V={voltage:.2f}V: {pct:.1f}%")

    print("\nLayer 2 (fc2):")
    unique2 = torch.unique(quantized.fc2.weight.data).sort()[0]
    print(f"  Weight values → Voltages:")
    for v in unique2:
        voltage = base_voltage * (1 + v * alpha2)
        count = (quantized.fc2.weight.data == v).sum().item()
        pct = 100 * count / quantized.fc2.weight.data.numel()
        print(f"    w={v:.4f} → V={voltage:.2f}V: {pct:.1f}%")
    
    print("\nEvaluating quantized model...")
    quantized_acc = evaluate(quantized, test_loader, device, T)
    print(f"Quantized accuracy: {quantized_acc:.2f}%")
    
    accuracy_drop = baseline_acc - quantized_acc
    print(f"\nAccuracy drop: {accuracy_drop:.2f}%")
    print(f"Relative drop: {100*accuracy_drop/baseline_acc:.1f}%")

    # Calculate energy consumption for quantized model
    # Energy proportional to V^2 for AFeFET devices
    def calculate_hardware_energy(model, alpha_dict, base_voltage=4.5):
        """Calculate actual hardware energy based on voltage levels used"""
        voltage_energy = {}
        voltage_counts = {}

        for v in voltage_levels:
            voltage_energy[v] = v ** 2  # Energy ∝ V^2
            voltage_counts[v] = 0

        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'fc1' in name:
                    alpha = alpha_dict['fc1']
                elif 'fc2' in name:
                    alpha = alpha_dict['fc2']
                else:
                    continue

                # Calculate actual voltage for each weight
                voltages = base_voltage * (1 + param.data * alpha)

                # Count how many operations at each voltage level
                for v in voltage_levels:
                    count = (torch.abs(voltages - v) < 0.01).sum().item()
                    voltage_counts[v] += count

        # Total energy = sum of (energy_per_op × num_ops) at each voltage
        total_energy = sum(voltage_energy[v] * voltage_counts[v] for v in voltage_levels)
        total_ops = sum(voltage_counts.values())

        return total_energy / total_ops, voltage_counts, total_ops

    alpha_dict_quantized = {'fc1': alpha1, 'fc2': alpha2}

    avg_energy, voltage_dist, total_ops = calculate_hardware_energy(
        quantized, alpha_dict_quantized, base_voltage
    )

    # Calculate theoretical minimum (all weights at lowest voltage)
    min_energy = voltage_levels[0] ** 2
    # Calculate theoretical maximum (all weights at highest voltage)
    max_energy = voltage_levels[-1] ** 2

    print(f"\nHardware Energy Analysis (AFeFET):")
    print(f"Average energy per operation: {avg_energy:.3f} (normalized)")
    print(f"Theoretical minimum (all @{voltage_levels[0]}V): {min_energy:.3f}")
    print(f"Theoretical maximum (all @{voltage_levels[-1]}V): {max_energy:.3f}")
    print(f"\nVoltage distribution ({total_ops} total operations):")
    for v in voltage_levels:
        pct = 100 * voltage_dist[v] / total_ops
        energy_contrib = (v**2) * voltage_dist[v] / total_ops
        print(f"  {v}V: {pct:5.1f}% of ops, energy contribution: {energy_contrib:.3f}")
    
    results = {
        'baseline_accuracy': float(baseline_acc),
        'quantized_accuracy': float(quantized_acc),
        'accuracy_drop': float(accuracy_drop),
        'voltage_levels': voltage_levels,
        'base_voltage': base_voltage,
        'scaling_factors': {
            'fc1_alpha': float(alpha1),
            'fc2_alpha': float(alpha2)
        },
        'energy_analysis': {
            'avg_energy_per_op': float(avg_energy),
            'min_theoretical_energy': float(min_energy),
            'max_theoretical_energy': float(max_energy),
            'voltage_distribution': {f'{v}V': int(voltage_dist[v]) for v in voltage_levels},
            'total_operations': int(total_ops)
        },
        'method': 'voltage_aware_quantization'
    }
    
    os.makedirs('../results/metrics', exist_ok=True)
    with open('../results/metrics/snn_quantization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(quantized.state_dict(), '../results/models/quantized_snn_v2.pth')
    
    print("\n✓ Results saved!")

if __name__ == '__main__':
    main()