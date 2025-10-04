import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import joblib

sns.set_style('whitegrid')

# Load data
df = pd.read_csv('data/processed/ml_dataset.csv')
energy_model = joblib.load('results/models/energy_model.pkl')
acc_model = joblib.load('results/models/accuracy_model.pkl')

with open('results/optimization_results.json') as f:
    opt_results = json.load(f)

# ========== FIGURE 1: Pareto Frontier ==========
fig, ax = plt.subplots(figsize=(8, 6))

# Plot all points
ax.scatter(df['energy_pj'], df['accuracy']*100, alpha=0.3, s=20, label='All configs')

# Highlight optimal points
for name, config in opt_results.items():
    ax.scatter(config['energy_pj'], config['accuracy']*100, 
               s=200, marker='*', label=name.replace('_', ' ').title())

ax.set_xlabel('Energy (pJ)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Energy-Accuracy Trade-off (Pareto Frontier)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/pareto_frontier.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pareto_frontier.png")

# ========== FIGURE 2: Energy vs Pulse Width ==========
fig, ax = plt.subplots(figsize=(8, 6))

for v in df['voltage_v'].unique():
    df_v = df[df['voltage_v'] == v]
    ax.plot(df_v['pulse_width_us'], df_v['energy_pj'], 
            marker='o', label=f'{v}V', linewidth=2)

ax.set_xlabel('Pulse Width (μs)', fontsize=12)
ax.set_ylabel('Energy (pJ)', fontsize=12)
ax.set_title('Energy vs Pulse Width at Different Voltages', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/energy_vs_pulse.png', dpi=300, bbox_inches='tight')
print("✓ Saved: energy_vs_pulse.png")

# ========== FIGURE 3: Comparison Table ==========
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Scenario', 'Pulse Width (μs)', 'Voltage (V)', 'Energy (pJ)', 'Accuracy (%)'])
for name, config in opt_results.items():
    table_data.append([
        name.replace('_', ' ').title(),
        f"{config['pulse_width_us']:.2f}",
        f"{config['voltage_v']:.2f}",
        f"{config['energy_pj']:.3f}",
        f"{config['accuracy']*100:.2f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.2, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.savefig('results/figures/comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_table.png")

print("\n✓ All figures created!")