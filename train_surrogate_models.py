import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# Load data
df = pd.read_csv('data/processed/ml_dataset.csv')

# Features and targets
features = ['pulse_width_us', 'voltage_v']
X = df[features].values

# Train separate models for energy and accuracy
models = {}

# ========== ENERGY MODEL ==========
print("Training energy prediction model...")
y_energy = df['energy_pj'].values

X_train, X_test, y_train, y_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)

energy_model = RandomForestRegressor(n_estimators=100, random_state=42)
energy_model.fit(X_train, y_train)

y_pred = energy_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Energy Model - R²: {r2:.4f}, MAE: {mae:.4f} pJ")
models['energy'] = energy_model

# ========== ACCURACY MODEL ==========
print("\nTraining accuracy prediction model...")
y_acc = df['accuracy'].values

X_train, X_test, y_train, y_test = train_test_split(X, y_acc, test_size=0.2, random_state=42)

acc_model = RandomForestRegressor(n_estimators=100, random_state=42)
acc_model.fit(X_train, y_train)

y_pred = acc_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Accuracy Model - R²: {r2:.4f}, MAE: {mae:.4f}")
models['accuracy'] = acc_model

# Save models
os.makedirs('results/models', exist_ok=True)
for name, model in models.items():
    joblib.dump(model, f'results/models/{name}_model.pkl')

print("\n✓ Models saved!")