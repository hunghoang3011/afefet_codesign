"""
Comparative Study: Perfect-Device Training vs Variability-Aware Training

Trains two models:
1. Baseline: Trained on perfect devices (no variation)
2. Robust: Trained with realistic device variations

Then tests both under increasing variability to quantify robustness improvement.

Key metrics:
- Accuracy degradation under variability
- Robustness gain (%)
- Worst-case performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional
import json
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from afefet_snn.variability_models import create_variability_scenarios, VariabilityModel
from train_variability_aware import VariabilityAwareSNN, train_epoch, test


def train_model(scenario_name, variability_model, epochs, device, train_loader, test_loader, lr=1e-3):
    """Train a model with given variability scenario"""

    model = VariabilityAwareSNN(
        variability_model=variability_model,
        T=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nTraining model: {scenario_name}")
    print(f"Variability: {variability_model.get_summary()}")

    best_acc = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

        scheduler.step()

    return model, best_acc, history


def evaluate_robustness(model, test_loader, device, variability_scenarios):
    """
    Evaluate model robustness under different variability conditions

    Returns:
        dict: {scenario_name: accuracy}
    """
    results = {}

    print("\nEvaluating robustness...")
    for scenario_name, variability_model in variability_scenarios.items():
        _, acc = test(model, device, test_loader, variability_scenario=variability_model)
        results[scenario_name] = acc
        print(f"  {scenario_name:20s}: {acc:.2f}%")

    return results


def create_severity_sweep():
    """
    Create variability scenarios with increasing severity for sweep analysis

    Returns:
        dict: {severity_level: VariabilityModel}
    """
    severities = {}

    # Level 0: Perfect (baseline)
    severities['0_perfect'] = VariabilityModel(
        enable_d2d=False, enable_c2c=False, enable_temp=False
    )

    # Level 1: Very low (3% D2D)
    severities['1_very_low'] = VariabilityModel(
        enable_d2d=True, enable_c2c=True,
        d2d_V_coercive_std=0.03, d2d_capacitance_std=0.02, d2d_area_ratio_std=0.03
    )

    # Level 2: Low (5% D2D)
    severities['2_low'] = VariabilityModel(
        enable_d2d=True, enable_c2c=True,
        d2d_V_coercive_std=0.05, d2d_capacitance_std=0.03, d2d_area_ratio_std=0.05
    )

    # Level 3: Moderate (10% D2D) - realistic
    severities['3_moderate'] = VariabilityModel(
        enable_d2d=True, enable_c2c=True,
        d2d_V_coercive_std=0.10, d2d_capacitance_std=0.05, d2d_area_ratio_std=0.08
    )

    # Level 4: High (15% D2D)
    severities['4_high'] = VariabilityModel(
        enable_d2d=True, enable_c2c=True,
        d2d_V_coercive_std=0.15, d2d_capacitance_std=0.10, d2d_area_ratio_std=0.12
    )

    # Level 5: Very high (20% D2D) - worst case
    severities['5_very_high'] = VariabilityModel(
        enable_d2d=True, enable_c2c=True,
        d2d_V_coercive_std=0.20, d2d_capacitance_std=0.15, d2d_area_ratio_std=0.15
    )

    # Level 6: Extreme (25% D2D) - beyond realistic
    severities['6_extreme'] = VariabilityModel(
        enable_d2d=True, enable_c2c=True,
        d2d_V_coercive_std=0.25, d2d_capacitance_std=0.20, d2d_area_ratio_std=0.20
    )

    return severities


def main():
    # Configuration
    EPOCHS = 50
    BATCH_SIZE = 256
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Get variability scenarios
    scenarios = create_variability_scenarios()

    # ========================================
    # EXPERIMENT 1: Train Perfect vs Robust
    # ========================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Perfect Device vs Variability-Aware Training")
    print("="*60)

    # Train model on perfect devices
    model_perfect, acc_perfect, history_perfect = train_model(
        'perfect', scenarios['perfect'], EPOCHS, DEVICE, train_loader, test_loader, LR
    )

    # Train model with moderate variability
    model_robust, acc_robust, history_robust = train_model(
        'd2d_moderate', scenarios['d2d_moderate'], EPOCHS, DEVICE, train_loader, test_loader, LR
    )

    print(f"\nBest Training Accuracy:")
    print(f"  Perfect Model: {acc_perfect:.2f}%")
    print(f"  Robust Model:  {acc_robust:.2f}%")

    # ========================================
    # EXPERIMENT 2: Robustness Evaluation
    # ========================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Robustness Under Standard Scenarios")
    print("="*60)

    print("\nPerfect Model:")
    robustness_perfect = evaluate_robustness(model_perfect, test_loader, DEVICE, scenarios)

    print("\nRobust Model:")
    robustness_robust = evaluate_robustness(model_robust, test_loader, DEVICE, scenarios)

    # ========================================
    # EXPERIMENT 3: Severity Sweep
    # ========================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Variability Severity Sweep")
    print("="*60)

    severity_scenarios = create_severity_sweep()

    print("\nPerfect Model (trained without variability):")
    sweep_perfect = evaluate_robustness(model_perfect, test_loader, DEVICE, severity_scenarios)

    print("\nRobust Model (trained with variability):")
    sweep_robust = evaluate_robustness(model_robust, test_loader, DEVICE, severity_scenarios)

    # ========================================
    # Analysis
    # ========================================
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Compute robustness improvement
    improvements = {}
    for scenario in scenarios.keys():
        acc_p = robustness_perfect[scenario]
        acc_r = robustness_robust[scenario]
        improvement = acc_r - acc_p
        improvements[scenario] = {
            'perfect_acc': acc_p,
            'robust_acc': acc_r,
            'improvement': improvement,
            'relative_improvement': (improvement / acc_p * 100) if acc_p > 0 else 0
        }

    print("\nRobustness Improvement (Robust - Perfect):")
    print(f"{'Scenario':20s} {'Perfect':>10s} {'Robust':>10s} {'Δ Acc':>10s} {'Relative':>10s}")
    print("-" * 65)
    for scenario, metrics in improvements.items():
        print(f"{scenario:20s} {metrics['perfect_acc']:>9.2f}% {metrics['robust_acc']:>9.2f}% "
              f"{metrics['improvement']:>9.2f}% {metrics['relative_improvement']:>9.2f}%")

    # Worst-case degradation
    worst_perfect = min(robustness_perfect.values())
    worst_robust = min(robustness_robust.values())

    print(f"\nWorst-Case Performance:")
    print(f"  Perfect Model: {worst_perfect:.2f}%")
    print(f"  Robust Model:  {worst_robust:.2f}%")
    print(f"  Improvement:   {worst_robust - worst_perfect:.2f}%")

    # Average degradation under variability
    avg_perfect = np.mean([v for k, v in robustness_perfect.items() if k != 'perfect'])
    avg_robust = np.mean([v for k, v in robustness_robust.items() if k != 'perfect'])

    print(f"\nAverage Performance (excluding perfect):")
    print(f"  Perfect Model: {avg_perfect:.2f}%")
    print(f"  Robust Model:  {avg_robust:.2f}%")
    print(f"  Improvement:   {avg_robust - avg_perfect:.2f}%")

    # ========================================
    # Save Results
    # ========================================
    results = {
        'experiment_config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'seed': SEED
        },
        'training_results': {
            'perfect': {
                'best_acc': acc_perfect,
                'history': history_perfect
            },
            'robust': {
                'best_acc': acc_robust,
                'history': history_robust
            }
        },
        'robustness_evaluation': {
            'perfect_model': robustness_perfect,
            'robust_model': robustness_robust,
            'improvements': improvements
        },
        'severity_sweep': {
            'perfect_model': sweep_perfect,
            'robust_model': sweep_robust
        },
        'summary': {
            'worst_case': {
                'perfect': worst_perfect,
                'robust': worst_robust,
                'improvement': worst_robust - worst_perfect
            },
            'average': {
                'perfect': avg_perfect,
                'robust': avg_robust,
                'improvement': avg_robust - avg_perfect
            }
        }
    }

    os.makedirs('results/variability', exist_ok=True)
    with open('results/variability/comparison_perfect_vs_robust.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save models
    os.makedirs('results/models', exist_ok=True)
    torch.save(model_perfect.state_dict(), 'results/models/model_perfect.pth')
    torch.save(model_robust.state_dict(), 'results/models/model_robust.pth')

    print(f"\n{'='*60}")
    print("Results saved to:")
    print("  - results/variability/comparison_perfect_vs_robust.json")
    print("  - results/models/model_perfect.pth")
    print("  - results/models/model_robust.pth")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
