import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('../results/metrics/baseline_v2_results.json') as f:
    baseline = json.load(f)

with open('../results/metrics/qat_snn_results.json') as f:
    qat = json.load(f)

with open('../results/metrics/afefet_full_results.json') as f:
    full = json.load(f)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

epochs_baseline = list(range(1, len(baseline['test_acc']) + 1))
epochs_qat = list(range(1, len(qat['history']['test_acc']) + 1))
epochs_full = list(range(1, len(full['history']['test_acc']) + 1))

ax.plot(epochs_baseline, baseline['test_acc'], 'o-', label='Baseline SNN', linewidth=2)
ax.plot(epochs_qat, qat['history']['test_acc'], 's-', label='QAT', linewidth=2)
ax.plot(epochs_full, full['history']['test_acc'], '^-', label='Full Device Physics', linewidth=2)

# Post-training quantization (single point)
ax.axhline(y=95.17, color='red', linestyle='--', label='Post-training Quant', linewidth=2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Training Convergence Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([85, 100])

plt.tight_layout()
plt.savefig('../results/figures/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_curves.png")
plt.close()

# Figure 2: Final comparison bar chart
methods = ['Baseline\nSNN', 'Post-training\nQuant', 'QAT', 'Full Device\nPhysics']
accuracies = [
    baseline['test_acc'][-1],
    95.17,
    qat['best_test_accuracy'],
    full['best_test_accuracy']
]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], width=0.6)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([90, 100])

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved accuracy_comparison.png")
plt.close()

# Figure 3: Summary table as image
import pandas as pd

data = {
    'Method': methods,
    'Accuracy (%)': [f"{a:.2f}" for a in accuracies],
    'Hardware\nCompatible': ['No', 'Yes', 'Yes', 'Yes'],
    'Device\nPhysics': ['No', 'No', 'No', 'Yes'],
    'Training\nTime': ['40 min', '+3 min', '40 min', '40 min']
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns,
                cellLoc='center', loc='center', 
                colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.savefig('../results/figures/comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved comparison_table.png")

print("\n✓✓✓ All figures created! Ready for presentation.")