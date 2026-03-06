"""
Generate 500 synthetic samples from the laterite dataset using CTGAN.
"""

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ── 1. Load & clean ──────────────────────────────────────────────
data = pd.read_csv('imputed_missforest.csv')

# Drop index column
if 'Sl. No' in data.columns:
    data = data.drop(columns=['Sl. No'])

print(f"Original data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# ── 2. Build metadata ────────────────────────────────────────────
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Explicitly mark categorical columns
metadata.update_column(column_name='Location', sdtype='categorical')
metadata.update_column(column_name='Soil Classification', sdtype='categorical')

# Explicitly mark numerical columns
numerical_columns = [
    'Specific Gravity', 'Gravel %', 'Sand %', 'Silt %', 'Clay %',
    'Liquid Limit %', 'Plastic Limit %', 'Plasticity Index %',
    'OMC %', 'MDD kN/m3', 'CBR % UnSoaked', 'CBR % Soaked', 'wPI'
]
for col in numerical_columns:
    if col in data.columns:
        metadata.update_column(column_name=col, sdtype='numerical')

print("\nMetadata validated successfully.")

# ── 3. Train CTGAN ───────────────────────────────────────────────
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=500,
    batch_size=50,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    verbose=True
)

print("\nTraining CTGAN (500 epochs)...")
synthesizer.fit(data)
print("Training complete!")

# ── 4. Generate 500 synthetic samples ────────────────────────────
synthetic_data = synthesizer.sample(num_rows=500)
print(f"\nSynthetic data shape: {synthetic_data.shape}")

# ── 5. Quick quality check ───────────────────────────────────────
print("\n── Statistical Comparison ──")
print(f"{'Column':<25} {'Real Mean':>12} {'Synth Mean':>12} {'Real Std':>12} {'Synth Std':>12}")
print("-" * 75)
for col in numerical_columns:
    if col in data.columns:
        r_mean = data[col].mean()
        s_mean = synthetic_data[col].mean()
        r_std = data[col].std()
        s_std = synthetic_data[col].std()
        print(f"{col:<25} {r_mean:>12.3f} {s_mean:>12.3f} {r_std:>12.3f} {s_std:>12.3f}")

print("\n── Categorical Distribution ──")
for col in ['Location', 'Soil Classification']:
    if col in data.columns:
        print(f"\n{col}:")
        real_dist = data[col].value_counts(normalize=True).head(8)
        synth_dist = synthetic_data[col].value_counts(normalize=True).head(8)
        print(f"  Real top categories:  {dict(real_dist)}")
        print(f"  Synth top categories: {dict(synth_dist)}")

# ── 6. Save ──────────────────────────────────────────────────────
output_path = 'ctgan_synthetic_500.csv'
synthetic_data.to_csv(output_path, index=False)
print(f"\n✓ Saved {len(synthetic_data)} synthetic samples to '{output_path}'")
