import numpy as np
from scipy.optimize import differential_evolution
import joblib
import json

# Load models
energy_model = joblib.load('results/models/energy_model.pkl')
acc_model = joblib.load('results/models/accuracy_model.pkl')

def predict_performance(pw, v):
    """Predict energy and accuracy"""
    X = np.array([[pw, v]])
    energy = energy_model.predict(X)[0]
    acc = acc_model.predict(X)[0]
    return energy, acc

# Grid search approach (more reliable than scipy minimize)
pulse_widths = np.logspace(0, 4, 50)  # 1 to 10000 μs
voltages = np.linspace(3.5, 5.5, 5)

# Evaluate all combinations
results_grid = []
for pw in pulse_widths:
    for v in voltages:
        energy, acc = predict_performance(pw, v)
        results_grid.append({
            'pw': pw,
            'v': v,
            'energy': energy,
            'acc': acc,
            'score_low_power': energy if acc >= 0.90 else 1e6,
            'score_balanced': 0.5*energy + 0.5*(1-acc)*1000 if acc >= 0.93 else 1e6,
            'score_high_acc': (1-acc)*1000 if energy <= 50 else 1e6
        })

import pandas as pd
df_grid = pd.DataFrame(results_grid)

# Find optimal for each scenario
scenarios = {}

# Low power: minimize energy with acc >= 90%
idx = df_grid['score_low_power'].idxmin()
scenarios['low_power'] = {
    'pulse_width_us': df_grid.loc[idx, 'pw'],
    'voltage_v': df_grid.loc[idx, 'v'],
    'energy_pj': df_grid.loc[idx, 'energy'],
    'accuracy': df_grid.loc[idx, 'acc']
}

# Balanced: minimize weighted sum
idx = df_grid['score_balanced'].idxmin()
scenarios['balanced'] = {
    'pulse_width_us': df_grid.loc[idx, 'pw'],
    'voltage_v': df_grid.loc[idx, 'v'],
    'energy_pj': df_grid.loc[idx, 'energy'],
    'accuracy': df_grid.loc[idx, 'acc']
}

# High accuracy: maximize accuracy with energy <= 50pJ
idx = df_grid['score_high_acc'].idxmin()
scenarios['high_accuracy'] = {
    'pulse_width_us': df_grid.loc[idx, 'pw'],
    'voltage_v': df_grid.loc[idx, 'v'],
    'energy_pj': df_grid.loc[idx, 'energy'],
    'accuracy': df_grid.loc[idx, 'acc']
}

# Print results
for name, config in scenarios.items():
    print(f"\n{'='*50}")
    print(f"{name.upper()}")
    print(f"{'='*50}")
    print(f"Pulse width: {config['pulse_width_us']:.2f} μs")
    print(f"Voltage: {config['voltage_v']:.2f} V")
    print(f"Energy: {config['energy_pj']:.3f} pJ")
    print(f"Accuracy: {config['accuracy']*100:.2f}%")

# Save
with open('results/optimization_results.json', 'w') as f:
    json.dump(scenarios, f, indent=2)

print("\n✓ Optimization complete!")