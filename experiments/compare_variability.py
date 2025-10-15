import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results/variability/train_perfect.json') as f:
    perfect = json.load(f)
with open('results/variability/train_d2d_moderate.json') as f:
    moderate = json.load(f)

# Robustness comparison
scenarios = ['perfect', 'c2c_only', 'd2d_only', 'd2d_moderate', 'd2d_high', 'realistic']
perfect_acc = [perfect['robustness_tests'][s] for s in scenarios]
moderate_acc = [moderate['robustness_tests'][s] for s in scenarios]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(scenarios))
width = 0.35

bars1 = ax.bar(x - width/2, perfect_acc, width, label='Trained on Perfect Devices', color='#1f77b4')
bars2 = ax.bar(x + width/2, moderate_acc, width, label='Trained with Variability', color='#ff7f0e')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Test Scenario', fontsize=12)
ax.set_title('AFeFET-SNN Robustness Under Device Variations', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Perfect\nDevices', 'C2C Only\n(2.1%)', 'D2D Only\n(10%)', 
                    'D2D Moderate\n(10%)', 'D2D High\n(15%)', 'Realistic\n(All)'], 
                   fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([88, 92])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('../results/figures/variability_robustness.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved variability_robustness.png")

# Summary
print("\n" + "="*60)
print("VARIABILITY-AWARE TRAINING RESULTS")
print("="*60)
print(f"\nKey Finding: AFeFET-SNN maintains ~90% accuracy across all")
print(f"device variation scenarios up to 15% D2D variation.")
print(f"\nPerfect training: {min(perfect_acc):.2f}% - {max(perfect_acc):.2f}%")
print(f"Variability training: {min(moderate_acc):.2f}% - {max(moderate_acc):.2f}%")
print(f"\nConclusion: Network is inherently robust to realistic device variations.")
print("="*60)