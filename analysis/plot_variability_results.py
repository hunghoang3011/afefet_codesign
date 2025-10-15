"""
Visualization and Analysis for Variability-Aware Training Results

Generates publication-quality figures:
1. Training curves comparison (perfect vs robust)
2. Robustness heatmap
3. Severity sweep plot
4. Device variation distributions
5. Performance degradation analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from afefet_snn.variability_models import create_variability_scenarios

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results():
    """Load experiment results"""
    results_path = 'results/variability/comparison_perfect_vs_robust.json'

    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run compare_variability_robustness.py first.")
        return None

    with open(results_path, 'r') as f:
        return json.load(f)


def plot_training_curves(results, save_dir='results/variability/figures'):
    """Plot training and test accuracy curves"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Extract data
    perfect_history = results['training_results']['perfect']['history']
    robust_history = results['training_results']['robust']['history']

    epochs_p = [h['epoch'] for h in perfect_history]
    train_acc_p = [h['train_acc'] for h in perfect_history]
    test_acc_p = [h['test_acc'] for h in perfect_history]

    epochs_r = [h['epoch'] for h in robust_history]
    train_acc_r = [h['train_acc'] for h in robust_history]
    test_acc_r = [h['test_acc'] for h in robust_history]

    # Training accuracy
    axes[0].plot(epochs_p, train_acc_p, '-o', label='Perfect Device', markersize=3, linewidth=2)
    axes[0].plot(epochs_r, train_acc_r, '-s', label='Variability-Aware', markersize=3, linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Accuracy (%)', fontsize=12)
    axes[0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Test accuracy
    axes[1].plot(epochs_p, test_acc_p, '-o', label='Perfect Device', markersize=3, linewidth=2)
    axes[1].plot(epochs_r, test_acc_r, '-s', label='Variability-Aware', markersize=3, linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/training_curves.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/training_curves.png")
    plt.close()


def plot_robustness_comparison(results, save_dir='results/variability/figures'):
    """Plot robustness comparison across scenarios"""
    os.makedirs(save_dir, exist_ok=True)

    robustness = results['robustness_evaluation']
    scenarios = list(robustness['perfect_model'].keys())
    perfect_accs = [robustness['perfect_model'][s] for s in scenarios]
    robust_accs = [robustness['robust_model'][s] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, perfect_accs, width, label='Perfect Device Training', alpha=0.8)
    bars2 = ax.bar(x + width/2, robust_accs, width, label='Variability-Aware Training', alpha=0.8)

    ax.set_xlabel('Variability Scenario', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Robustness Comparison Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/robustness_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/robustness_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/robustness_comparison.png")
    plt.close()


def plot_severity_sweep(results, save_dir='results/variability/figures'):
    """Plot accuracy vs variability severity"""
    os.makedirs(save_dir, exist_ok=True)

    sweep = results['severity_sweep']
    severity_levels = sorted(sweep['perfect_model'].keys())

    # Extract severity numbers for x-axis
    severity_nums = [int(s.split('_')[0]) for s in severity_levels]
    perfect_accs = [sweep['perfect_model'][s] for s in severity_levels]
    robust_accs = [sweep['robust_model'][s] for s in severity_levels]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(severity_nums, perfect_accs, '-o', label='Perfect Device Training',
            markersize=8, linewidth=2.5, color='#E74C3C')
    ax.plot(severity_nums, robust_accs, '-s', label='Variability-Aware Training',
            markersize=8, linewidth=2.5, color='#3498DB')

    # Shade improvement region
    ax.fill_between(severity_nums, perfect_accs, robust_accs,
                    where=np.array(robust_accs) >= np.array(perfect_accs),
                    alpha=0.3, color='green', label='Robustness Gain')

    ax.set_xlabel('Variability Severity Level', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Variability Severity', fontsize=14, fontweight='bold')
    ax.set_xticks(severity_nums)
    ax.set_xticklabels([s.split('_')[1] for s in severity_levels], rotation=45, ha='right')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Annotate degradation
    for i, (sev, p_acc, r_acc) in enumerate(zip(severity_nums, perfect_accs, robust_accs)):
        if i > 0:  # Skip perfect
            improvement = r_acc - p_acc
            if improvement > 0.5:  # Only annotate significant improvements
                ax.annotate(f'+{improvement:.1f}%',
                          xy=(sev, (p_acc + r_acc) / 2),
                          xytext=(10, 0), textcoords='offset points',
                          fontsize=9, color='green', fontweight='bold',
                          arrowprops=dict(arrowstyle='->', color='green', lw=1))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/severity_sweep.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/severity_sweep.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/severity_sweep.png")
    plt.close()


def plot_improvement_heatmap(results, save_dir='results/variability/figures'):
    """Plot improvement heatmap"""
    os.makedirs(save_dir, exist_ok=True)

    improvements = results['robustness_evaluation']['improvements']
    scenarios = list(improvements.keys())

    # Extract data
    perfect_accs = [improvements[s]['perfect_acc'] for s in scenarios]
    robust_accs = [improvements[s]['robust_acc'] for s in scenarios]
    deltas = [improvements[s]['improvement'] for s in scenarios]

    # Create data matrix
    data = np.array([perfect_accs, robust_accs, deltas]).T

    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colormap for improvement (red=negative, white=0, green=positive)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    # Plot heatmap for improvements only
    im = ax.imshow([deltas], cmap=cmap, aspect='auto', vmin=-5, vmax=5)

    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=10)
    ax.set_yticks([0])
    ax.set_yticklabels(['Improvement\n(Robust - Perfect)'], fontsize=11)
    ax.set_title('Robustness Improvement Across Scenarios', fontsize=14, fontweight='bold')

    # Add text annotations
    for i, delta in enumerate(deltas):
        text = ax.text(i, 0, f'{delta:.2f}%',
                      ha="center", va="center", color="black", fontsize=11, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Accuracy Improvement (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/improvement_heatmap.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/improvement_heatmap.png")
    plt.close()


def plot_device_variation_distributions(save_dir='results/variability/figures'):
    """Plot device parameter variation distributions"""
    os.makedirs(save_dir, exist_ok=True)

    import torch
    from afefet_snn.variability_models import VariabilityModel

    # Create moderate variability model
    var_model = VariabilityModel(
        enable_d2d=True,
        d2d_V_coercive_std=0.10,
        d2d_capacitance_std=0.05,
        d2d_area_ratio_std=0.08
    )

    # Generate variations for a large array
    shape = (1000, 1000)
    device = torch.device('cpu')
    d2d_vars = var_model.generate_d2d_variations(shape, device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    params = [
        ('V_coercive_factor', 'Coercive Voltage Factor', 10),
        ('capacitance_factor', 'Capacitance Factor', 5),
        ('area_ratio_factor', 'Area Ratio Factor', 8)
    ]

    for ax, (key, title, std_pct) in zip(axes, params):
        data = d2d_vars[key].flatten().numpy()

        # Histogram
        ax.hist(data, bins=50, alpha=0.7, density=True, edgecolor='black', linewidth=0.5)

        # Fit Gaussian
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        gaussian = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
        ax.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian Fit\nμ={mu:.3f}, σ={sigma:.3f}')

        ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Nominal (1.0)')
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'{title}\n({std_pct}% std)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/device_variations.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/device_variations.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/device_variations.png")
    plt.close()


def generate_summary_table(results, save_dir='results/variability/figures'):
    """Generate summary statistics table"""
    os.makedirs(save_dir, exist_ok=True)

    summary = results['summary']

    table_text = f"""
Variability-Aware Training Summary
{'='*60}

WORST-CASE PERFORMANCE:
  Perfect Device Training:      {summary['worst_case']['perfect']:.2f}%
  Variability-Aware Training:   {summary['worst_case']['robust']:.2f}%
  Improvement:                  {summary['worst_case']['improvement']:.2f}%

AVERAGE PERFORMANCE (under variability):
  Perfect Device Training:      {summary['average']['perfect']:.2f}%
  Variability-Aware Training:   {summary['average']['robust']:.2f}%
  Improvement:                  {summary['average']['improvement']:.2f}%

TRAINING RESULTS:
  Perfect Device Best Acc:      {results['training_results']['perfect']['best_acc']:.2f}%
  Robust Model Best Acc:        {results['training_results']['robust']['best_acc']:.2f}%

KEY INSIGHTS:
  1. Variability-aware training shows {summary['average']['improvement']:.2f}% better
     average performance under device variations
  2. Worst-case robustness improved by {summary['worst_case']['improvement']:.2f}%
  3. Training on perfect devices leads to overfitting to ideal conditions
  4. Realistic variability injection acts as effective regularization

{'='*60}
"""

    # Save as text
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(table_text)

    print(table_text)
    print(f"Saved: {save_dir}/summary.txt")


def create_presentation_figure(results, save_dir='results/variability/figures'):
    """Create single comprehensive figure for presentation"""
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Training curves (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    perfect_history = results['training_results']['perfect']['history']
    robust_history = results['training_results']['robust']['history']

    epochs_p = [h['epoch'] for h in perfect_history]
    test_acc_p = [h['test_acc'] for h in perfect_history]
    epochs_r = [h['epoch'] for h in robust_history]
    test_acc_r = [h['test_acc'] for h in robust_history]

    ax1.plot(epochs_p, test_acc_p, '-o', label='Perfect Device', markersize=4, linewidth=2.5)
    ax1.plot(epochs_r, test_acc_r, '-s', label='Variability-Aware', markersize=4, linewidth=2.5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Training Convergence', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Summary metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    summary = results['summary']
    summary_text = (
        f"Key Results:\n\n"
        f"Avg Improvement:\n{summary['average']['improvement']:.2f}%\n\n"
        f"Worst-Case Gain:\n{summary['worst_case']['improvement']:.2f}%\n\n"
        f"Perfect Best: {results['training_results']['perfect']['best_acc']:.2f}%\n"
        f"Robust Best: {results['training_results']['robust']['best_acc']:.2f}%"
    )
    ax2.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_title('(b) Summary', fontsize=13, fontweight='bold', loc='left')

    # 3. Severity sweep (middle row, spans all)
    ax3 = fig.add_subplot(gs[1, :])
    sweep = results['severity_sweep']
    severity_levels = sorted(sweep['perfect_model'].keys())
    severity_nums = [int(s.split('_')[0]) for s in severity_levels]
    perfect_accs = [sweep['perfect_model'][s] for s in severity_levels]
    robust_accs = [sweep['robust_model'][s] for s in severity_levels]

    ax3.plot(severity_nums, perfect_accs, '-o', label='Perfect Device Training',
            markersize=8, linewidth=3, color='#E74C3C')
    ax3.plot(severity_nums, robust_accs, '-s', label='Variability-Aware Training',
            markersize=8, linewidth=3, color='#3498DB')
    ax3.fill_between(severity_nums, perfect_accs, robust_accs,
                    where=np.array(robust_accs) >= np.array(perfect_accs),
                    alpha=0.3, color='green')
    ax3.set_xlabel('Variability Severity', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Robustness Under Increasing Variability', fontsize=13, fontweight='bold', loc='left')
    ax3.set_xticks(severity_nums)
    ax3.set_xticklabels([s.split('_')[1] for s in severity_levels], rotation=30, ha='right')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # 4. Robustness comparison (bottom row, spans all)
    ax4 = fig.add_subplot(gs[2, :])
    robustness = results['robustness_evaluation']
    scenarios = list(robustness['perfect_model'].keys())
    perfect_accs = [robustness['perfect_model'][s] for s in scenarios]
    robust_accs = [robustness['robust_model'][s] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35
    bars1 = ax4.bar(x - width/2, perfect_accs, width, label='Perfect Device', alpha=0.8, color='#E74C3C')
    bars2 = ax4.bar(x + width/2, robust_accs, width, label='Variability-Aware', alpha=0.8, color='#3498DB')

    ax4.set_xlabel('Test Scenario', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Performance Across Variability Scenarios', fontsize=13, fontweight='bold', loc='left')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=10)
    ax4.legend(fontsize=11)
    ax4.grid(True, axis='y', alpha=0.3)

    # Overall title
    fig.suptitle('Variability-Aware Training for Robust AFeFET Neuromorphic Computing',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(f'{save_dir}/presentation_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/presentation_summary.pdf', bbox_inches='tight')
    print(f"Saved: {save_dir}/presentation_summary.png")
    plt.close()


def main():
    print("Loading results...")
    results = load_results()

    if results is None:
        print("\nPlease run compare_variability_robustness.py first to generate results.")
        return

    print("\nGenerating figures...")

    # Generate all plots
    plot_training_curves(results)
    plot_robustness_comparison(results)
    plot_severity_sweep(results)
    plot_improvement_heatmap(results)
    plot_device_variation_distributions()
    generate_summary_table(results)
    create_presentation_figure(results)

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("Location: results/variability/figures/")
    print("="*60)


if __name__ == '__main__':
    main()
