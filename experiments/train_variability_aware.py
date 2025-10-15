"""
Variability-Aware Training for AFeFET-SNN

Trains SNN with realistic device variations injected during training.
Compares robustness against perfect-device baseline.

Key contributions:
1. Device-to-device (D2D) variation: V_c, capacitance, area ratio
2. Cycle-to-cycle (C2C) variation: Write/read noise
3. Temperature variation: ±20°C operating range
4. Retention variation: Time constant uncertainty

Expected outcome:
- Models trained with variability show better robustness
- Graceful degradation under severe variations
- Quantify AI tolerance to manufacturing imperfections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate
import json
import os
import sys
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from afefet_snn.device_physics import MFMISDevice, TemporalDynamics
from afefet_snn.variability_models import (
    VariabilityModel, VariabilityAwareDevice, create_variability_scenarios
)


class VariabilityAwareAFeFETQuantizer(torch.autograd.Function):
    """
    Voltage quantizer with variability injection
    """
    @staticmethod
    def forward(ctx, weights, alpha, voltage_levels, base_voltage, variability_device):
        # Voltage mapping
        target_voltages = base_voltage * (1 + weights * alpha)

        # Quantize
        v_expanded = target_voltages.unsqueeze(-1)
        distances = torch.abs(v_expanded - voltage_levels)
        indices = torch.argmin(distances, dim=-1)
        quantized_voltages = voltage_levels[indices]

        # Apply write noise with variability (C2C + D2D effects)
        quantized_voltages = variability_device.write_noise(quantized_voltages)

        # Back to weight space
        quantized_weights = (quantized_voltages / base_voltage - 1) / alpha

        return quantized_weights

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class VariabilityAwareAFeFETLinear(nn.Module):
    """
    AFeFET linear layer with variability injection during training
    """
    def __init__(
        self,
        in_features,
        out_features,
        voltage_levels=[3.5, 4.0, 4.5, 5.0, 5.5],
        base_voltage=4.5,
        area_ratio=1.0,
        mode='LTM',
        variability_model=None,
        layer_id="layer",
        bias=False
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.voltage_levels = torch.tensor(voltage_levels)
        self.base_voltage = base_voltage
        self.mode = mode
        self.layer_id = layer_id

        # Base device model
        base_device = MFMISDevice(
            area_ratio=area_ratio,
            base_voltage=base_voltage,
            stm_retention=0.3,
            temperature=300.0
        )

        # Wrap with variability if provided
        if variability_model is not None:
            self.device = VariabilityAwareDevice(
                base_device, variability_model, device_id=layer_id
            )
        else:
            self.device = base_device

        # Learnable parameters
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Device state tracking
        self.register_buffer('write_count', torch.zeros(out_features, in_features, dtype=torch.long))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, 0, 0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, apply_retention=False, time_elapsed=1.0):
        if self.voltage_levels.device != self.weight.device:
            self.voltage_levels = self.voltage_levels.to(self.weight.device)

        if self.training:
            # Training: apply variability
            quantized_weight = VariabilityAwareAFeFETQuantizer.apply(
                self.weight, self.alpha, self.voltage_levels,
                self.base_voltage, self.device
            )
            self.write_count += 1

        else:
            # Inference: quantize with optional variability
            with torch.no_grad():
                target_voltages = self.base_voltage * (1 + self.weight * self.alpha)
                v_expanded = target_voltages.unsqueeze(-1)
                distances = torch.abs(v_expanded - self.voltage_levels)
                indices = torch.argmin(distances, dim=-1)
                quantized_voltages = self.voltage_levels[indices]
                quantized_weight = (quantized_voltages / self.base_voltage - 1) / self.alpha

                # Apply read noise during inference
                if hasattr(self.device, 'read_noise'):
                    quantized_weight = self.device.read_noise(quantized_weight)

        return F.linear(x, quantized_weight, self.bias)


class VariabilityAwareSNN(nn.Module):
    """
    SNN with variability-aware AFeFET synapses
    """
    def __init__(
        self,
        voltage_levels=[3.5, 4.0, 4.5, 5.0, 5.5],
        area_ratio_layer1=1.0,
        area_ratio_layer2=1.5,
        variability_model=None,
        T=8
    ):
        super().__init__()

        self.T = T
        self.variability_model = variability_model

        # AFeFET layers with variability
        self.fc1 = VariabilityAwareAFeFETLinear(
            784, 256,
            voltage_levels=voltage_levels,
            area_ratio=area_ratio_layer1,
            variability_model=variability_model,
            layer_id="fc1"
        )
        self.lif1 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )

        self.fc2 = VariabilityAwareAFeFETLinear(
            256, 10,
            voltage_levels=voltage_levels,
            area_ratio=area_ratio_layer2,
            variability_model=variability_model,
            layer_id="fc2"
        )
        self.lif2 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Time-stepped spiking
        spikes_out = []
        for t in range(self.T):
            x_t = x if t == 0 else torch.zeros_like(x)
            x_t = self.fc1(x_t)
            x_t = self.lif1(x_t)
            x_t = self.fc2(x_t)
            x_t = self.lif2(x_t)
            spikes_out.append(x_t)

        # Sum over time
        out = torch.stack(spikes_out, dim=0).sum(0)
        return out


def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        # Reset spiking neurons
        functional.reset_net(model)

        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    return total_loss / len(train_loader), 100. * correct / total


def test(model, device, test_loader, variability_scenario=None):
    """
    Test model with optional variability injection

    Args:
        variability_scenario: If provided, inject this variability during inference
    """
    model.eval()
    test_loss = 0
    correct = 0

    # Temporarily swap variability model if testing with different scenario
    original_variability = {}
    if variability_scenario is not None:
        for name, module in model.named_modules():
            if isinstance(module, VariabilityAwareAFeFETLinear):
                original_variability[name] = module.device
                # Create new variability-aware device
                base_device = module.device.base_device if hasattr(module.device, 'base_device') else module.device
                module.device = VariabilityAwareDevice(
                    base_device, variability_scenario, device_id=module.layer_id
                )

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            functional.reset_net(model)

    # Restore original variability
    if variability_scenario is not None:
        for name, module in model.named_modules():
            if isinstance(module, VariabilityAwareAFeFETLinear):
                module.device = original_variability[name]

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='d2d_moderate',
                       choices=['perfect', 'c2c_only', 'd2d_only', 'd2d_moderate', 'd2d_high', 'realistic'],
                       help='Variability scenario for training')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create variability model
    scenarios = create_variability_scenarios()
    variability_model = scenarios[args.scenario]

    print(f"\nTraining with variability scenario: {args.scenario}")
    print(variability_model.get_summary())

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = VariabilityAwareSNN(
        variability_model=variability_model,
        T=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\n{'='*60}")
    print(f"Training Started")
    print(f"{'='*60}")

    best_acc = 0
    results = {
        'scenario': args.scenario,
        'variability_config': variability_model.get_summary(),
        'train_history': [],
        'test_history': [],
        'robustness_tests': {}
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)

        results['train_history'].append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
        results['test_history'].append({'epoch': epoch, 'loss': test_loss, 'acc': test_acc})

        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            os.makedirs('results/models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'scenario': args.scenario
            }, f'results/models/variability_aware_{args.scenario}_best.pth')

        scheduler.step()

    # Robustness evaluation: Test trained model under different variability conditions
    print(f"\n{'='*60}")
    print(f"Robustness Evaluation")
    print(f"{'='*60}")

    for test_scenario_name, test_variability in scenarios.items():
        print(f"\nTesting under {test_scenario_name}...")
        _, test_acc = test(model, device, test_loader, variability_scenario=test_variability)
        results['robustness_tests'][test_scenario_name] = test_acc
        print(f"  Accuracy: {test_acc:.2f}%")

    results['best_accuracy'] = best_acc

    # Save results
    os.makedirs('results/variability', exist_ok=True)
    with open(f'results/variability/train_{args.scenario}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Results saved to: results/variability/train_{args.scenario}.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
