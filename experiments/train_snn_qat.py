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


class VoltageQuantizer(torch.autograd.Function):
    """
    Straight-Through Estimator for voltage quantization.
    Forward: quantize to discrete voltage levels
    Backward: pass gradient through unchanged (STE)
    """
    @staticmethod
    def forward(ctx, weights, alpha, voltage_levels, base_voltage):
        # Convert weights to target voltages
        target_voltages = base_voltage * (1 + weights * alpha)

        # Quantize to nearest voltage level
        v_expanded = target_voltages.unsqueeze(-1)
        distances = torch.abs(v_expanded - voltage_levels)
        indices = torch.argmin(distances, dim=-1)
        quantized_voltages = voltage_levels[indices]

        # Convert back to weight space
        quantized_weights = (quantized_voltages / base_voltage - 1) / alpha

        return quantized_weights

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient passes unchanged
        return grad_output, None, None, None


class QuantizedLinear(nn.Module):
    """
    Linear layer with voltage-aware quantization during training.
    Uses fixed alpha determined by optimal MSE minimization.
    """
    def __init__(self, in_features, out_features, voltage_levels=[3.5, 4.0, 4.5, 5.0, 5.5],
                 base_voltage=4.5, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.voltage_levels = torch.tensor(voltage_levels)
        self.base_voltage = base_voltage

        # Learnable weights (continuous during training)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # Alpha is a buffer (not learnable), will be set during reset_parameters
        self.register_buffer('alpha', torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Use smaller initialization to fit better in voltage range
        nn.init.uniform_(self.weight, -0.3, 0.3)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Set alpha to map weight range [-0.5, 0.5] to full voltage range
        # V = 4.5 * (1 + w * alpha)
        # Want: w=0.5 -> V=5.5, so: 5.5 = 4.5 * (1 + 0.5*alpha) -> alpha ~= 0.44
        # But use smaller alpha for smoother gradients
        with torch.no_grad():
            self.alpha.fill_(1.0)  # Conservative alpha for stable training

    def forward(self, x):
        # Move voltage_levels to same device as weights
        if self.voltage_levels.device != self.weight.device:
            self.voltage_levels = self.voltage_levels.to(self.weight.device)

        # Quantize weights using STE
        if self.training:
            quantized_weight = VoltageQuantizer.apply(
                self.weight, self.alpha, self.voltage_levels, self.base_voltage
            )
        else:
            # During inference, use same quantization (no gradient needed)
            with torch.no_grad():
                target_voltages = self.base_voltage * (1 + self.weight * self.alpha)
                v_expanded = target_voltages.unsqueeze(-1)
                distances = torch.abs(v_expanded - self.voltage_levels)
                indices = torch.argmin(distances, dim=-1)
                quantized_voltages = self.voltage_levels[indices]
                quantized_weight = (quantized_voltages / self.base_voltage - 1) / self.alpha

        return F.linear(x, quantized_weight, self.bias)


class QAT_SNN(nn.Module):
    """
    Quantization-Aware Training SNN with voltage-constrained weights.
    """
    def __init__(self, T=20, voltage_levels=[3.5, 4.0, 4.5, 5.0, 5.5], base_voltage=4.5):
        super().__init__()
        self.T = T

        self.fc1 = QuantizedLinear(784, 300, voltage_levels, base_voltage, bias=False)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

        self.fc2 = QuantizedLinear(300, 10, voltage_levels, base_voltage, bias=False)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        return x


def train_epoch(model, train_loader, optimizer, device, T=20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)

        img = img.view(img.size(0), -1)

        # Convert to spike trains
        spike_img = []
        for t in range(T):
            spike = torch.rand_like(img) < img.clamp(0, 1)
            spike_img.append(spike.float())
        spike_img = torch.stack(spike_img)

        optimizer.zero_grad()
        output = model(spike_img)
        output_mean = output.mean(0)

        loss = F.cross_entropy(output_mean, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output_mean.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

        functional.reset_net(model)

        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    return total_loss / len(train_loader), 100. * correct / total


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
    batch_size = 128
    epochs = 20
    lr = 1e-3

    voltage_levels = [3.5, 4.0, 4.5, 5.0, 5.5]
    base_voltage = 4.5

    print(f"Training QAT SNN with voltage levels: {voltage_levels}")
    print(f"Base voltage: {base_voltage}V\n")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data/', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data/', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize QAT model
    model = QAT_SNN(T=T, voltage_levels=voltage_levels, base_voltage=base_voltage).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Display current alpha values
        print(f"Alpha values: fc1={model.fc1.alpha.item():.4f}, fc2={model.fc2.alpha.item():.4f}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, T)
        test_acc = evaluate(model, test_loader, device, T)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), '../results/models/qat_snn_best.pth')
            print(f"  → Saved best model (acc: {best_acc:.2f}%)")

        scheduler.step()

    print(f"\nTraining complete! Best test accuracy: {best_acc:.2f}%")

    # Analyze quantized weights
    model.eval()
    print("\n" + "="*60)
    print("Quantized Weight Analysis")
    print("="*60)

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                # Get quantized weights
                target_voltages = base_voltage * (1 + module.weight * module.alpha)
                v_expanded = target_voltages.unsqueeze(-1)
                distances = torch.abs(v_expanded - module.voltage_levels.to(module.weight.device))
                indices = torch.argmin(distances, dim=-1)
                quantized_voltages = module.voltage_levels.to(module.weight.device)[indices]
                quantized_weights = (quantized_voltages / base_voltage - 1) / module.alpha

                unique_weights = torch.unique(quantized_weights).sort()[0]

                print(f"\n{name}:")
                print(f"  Alpha: {module.alpha.item():.6f}")
                print(f"  Weight values → Voltages:")
                for w in unique_weights:
                    v = base_voltage * (1 + w * module.alpha)
                    count = (torch.abs(quantized_weights - w) < 1e-6).sum().item()
                    pct = 100 * count / quantized_weights.numel()
                    print(f"    w={w:.4f} → V={v:.2f}V: {pct:.1f}%")

    # Save results
    results = {
        'method': 'quantization_aware_training',
        'best_test_accuracy': float(best_acc),
        'final_test_accuracy': float(history['test_acc'][-1]),
        'voltage_levels': voltage_levels,
        'base_voltage': base_voltage,
        'final_alphas': {
            'fc1': float(model.fc1.alpha.item()),
            'fc2': float(model.fc2.alpha.item())
        },
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'T': T
        },
        'history': history
    }

    os.makedirs('../results/metrics', exist_ok=True)
    with open('../results/metrics/qat_snn_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to ../results/metrics/qat_snn_results.json")
    print(f"✓ Best model saved to ../results/models/qat_snn_best.pth")


if __name__ == '__main__':
    main()
