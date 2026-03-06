"""Verify the generated synthetic data."""

import pandas as pd
import numpy as np

# Load data
real_data = pd.read_csv('../imputed_missforest.csv')
synthetic_data = pd.read_csv('laterite_synthetic.csv')

print("=" * 70)
print("KDE-Copula GAN - Data Verification")
print("=" * 70)

print(f"\nReal Data Shape: {real_data.shape}")
print(f"Synthetic Data Shape: {synthetic_data.shape}")

print(f"\n{'Column':<25} {'Real Mean':<15} {'Synth Mean':<15} {'Real Std':<15} {'Synth Std':<15}")
print("-" * 85)

# Define numerical columns
numerical_columns = [
    'Specific Gravity', 'Gravel %', 'Sand %', 'Silt %', 'Clay %',
    'Liquid Limit %', 'Plastic Limit %', 'Plasticity Index %',
    'OMC %', 'MDD kN/m3', 'CBR % UnSoaked', 'CBR % Soaked', 'wPI'
]

for col in numerical_columns:
    if col in real_data.columns and col in synthetic_data.columns:
        real_mean = real_data[col].mean()
        synth_mean = synthetic_data[col].mean()
        real_std = real_data[col].std()
        synth_std = synthetic_data[col].std()
        print(f"{col:<25} {real_mean:<15.3f} {synth_mean:<15.3f} {real_std:<15.3f} {synth_std:<15.3f}")

print("\nCategorical Columns:")
print("-" * 70)
print(f"\nLocation:")
print(f"  Real unique values: {real_data['Location'].nunique()}")
print(f"  Synthetic unique values: {synthetic_data['Location'].nunique()}")
print(f"  Real values: {sorted(real_data['Location'].unique())}")
print(f"  Synthetic values: {sorted(synthetic_data['Location'].unique())}")

print(f"\nSoil Classification:")
print(f"  Real unique values: {real_data['Soil Classification'].nunique()}")
print(f"  Synthetic unique values: {synthetic_data['Soil Classification'].nunique()}")
print(f"  Real values: {sorted(real_data['Soil Classification'].unique())}")
print(f"  Synthetic values: {sorted(synthetic_data['Soil Classification'].unique())}")

print("\n" + "=" * 70)
print("✓ Verification Complete!")
print("=" * 70)
