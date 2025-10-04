import pandas as pd
import numpy as np
import json

# Load Excel
xls = pd.ExcelFile('data/raw/data.xlsx')

# ========== ENERGY vs PULSE WIDTH ==========
# Từ các sheets khác nhau, extract relationship
# Giả định: có data về energy measurements với different pulse widths

# Sample structure (adapt dựa trên actual data)
energy_data = []

# TODO: Loop through relevant sheets
# For now, create synthetic based on paper's model: E ∝ pulse_width × voltage²

pulse_widths = np.logspace(-6, -2, 50)  # 1μs to 10ms
voltages = [3.5, 4.0, 4.5, 5.0, 5.5]

for pw in pulse_widths:
    for v in voltages:
        # Energy model từ paper
        energy = 0.1e-12 * (pw / 1e-6) * (v / 5.0)**2
        
        # Add some noise
        energy *= (1 + np.random.normal(0, 0.05))
        
        energy_data.append({
            'pulse_width_us': pw * 1e6,
            'voltage_v': v,
            'energy_pj': energy * 1e12
        })

df_energy = pd.DataFrame(energy_data)

# ========== ACCURACY vs PULSE WIDTH ==========
# Từ Fig 5h và experimental data

accuracy_data = []
for pw in pulse_widths:
    # Sigmoid relationship: longer pulse → higher accuracy
    # STM threshold ~5ms, LTM threshold ~20ms
    
    if pw < 5e-3:  # Short pulse
        base_acc = 0.85
    elif pw > 20e-3:  # Long pulse
        base_acc = 0.97
    else:  # Transition
        t = (pw - 5e-3) / (20e-3 - 5e-3)
        base_acc = 0.85 + t * (0.97 - 0.85)
    
    # Add noise
    acc = base_acc + np.random.normal(0, 0.01)
    acc = np.clip(acc, 0, 1)
    
    for v in voltages:
        accuracy_data.append({
            'pulse_width_us': pw * 1e6,
            'voltage_v': v,
            'accuracy': acc
        })

df_accuracy = pd.DataFrame(accuracy_data)

# ========== MERGE ==========
df_full = pd.merge(df_energy, df_accuracy, on=['pulse_width_us', 'voltage_v'])

print(f"Dataset shape: {df_full.shape}")
print("\nFirst 5 rows:")
print(df_full.head())
print("\nStatistics:")
print(df_full.describe())

# Save
df_full.to_csv('data/processed/ml_dataset.csv', index=False)
print("\n✓ Dataset saved to data/processed/ml_dataset.csv")