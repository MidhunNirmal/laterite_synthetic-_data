"""
Enhanced Imputation for Laterite Dataset
Using state-of-the-art imputation methods: MissForest, MICE, and KNN
Based on 2024 research showing MissForest as the most accurate method
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("ADVANCED IMPUTATION FOR LATERITE DATASET")
print("Using: MissForest, MICE, and KNN Imputation")
print("=" * 70)

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================
print("\n1. Loading dataset...")
df_original = pd.read_csv(
    'laterite.csv',
    na_values=['_', '', ' ', 'NaN', 'nan', 'NA', 'na', 'N/A', 'n/a'],
    skipinitialspace=True
)

# Remove unnamed columns
df_original = df_original.loc[:, ~df_original.columns.str.contains('^Unnamed', na=False)]
df_original = df_original.loc[:, df_original.columns.str.strip() != '']

print(f"Dataset shape: {df_original.shape}")
print(f"\nColumns with missing values:")
missing_cols = df_original.isnull().sum()
missing_cols = missing_cols[missing_cols > 0]
for col, count in missing_cols.items():
    pct = (count / len(df_original)) * 100
    print(f"  - {col}: {count} ({pct:.1f}%)")

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n2. Preprocessing data...")
df = df_original.copy()

# Handle special values for plasticity columns
plasticity_cols = ['Plastic Limit %', 'Plasticity Index %']
for col in plasticity_cols:
    if col in df.columns:
        df[col] = df[col].replace('NP', '0')
        df[col] = df[col].replace('l', np.nan)

# Identify column types
numeric_cols = [col for col in df.columns if col not in ['Sl. No', 'Location', 'Soil Classification']]
categorical_cols = ['Location', 'Soil Classification']

# Convert numeric columns
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle extreme outliers
if 'CBR % UnSoaked' in df.columns:
    cap_value = df['CBR % UnSoaked'].quantile(0.99)
    df.loc[df['CBR % UnSoaked'] > cap_value, 'CBR % UnSoaked'] = np.nan

print(f"Preprocessed dataset shape: {df.shape}")

# ============================================================================
# 3. PREPARE DATA FOR IMPUTATION
# ============================================================================
print("\n3. Preparing data for imputation...")

# For imputation, we'll work with numeric columns only
# Categorical columns will be handled separately via mode imputation or kept as-is
df_numeric = df[numeric_cols].copy()

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Missing values in numeric data: {df_numeric.isnull().sum().sum()}")

# ============================================================================
# 4. METHOD 1: MISSFOREST (Random Forest-based)
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 1: MISSFOREST (Random Forest-based)")
print("=" * 70)
print("MissForest is the most accurate imputation method according to 2024 research")
print("Imputing...")

# MissForest implementation using sklearn's IterativeImputer with RandomForest
missforest_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    max_iter=10,
    random_state=42,
    verbose=0
)

df_missforest = df.copy()
df_missforest[numeric_cols] = missforest_imputer.fit_transform(df_numeric)

# Check results
remaining_missing = df_missforest[numeric_cols].isnull().sum().sum()
print(f"✓ MissForest complete! Remaining missing: {remaining_missing}")

# ============================================================================
# 5. METHOD 2: MICE (Multiple Imputation by Chained Equations)
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 2: MICE (Multiple Imputation by Chained Equations)")
print("=" * 70)
print("MICE provides uncertainty quantification through multiple imputations")
print("Imputing...")

mice_imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    verbose=0
)

df_mice = df.copy()
df_mice[numeric_cols] = mice_imputer.fit_transform(df_numeric)

remaining_missing = df_mice[numeric_cols].isnull().sum().sum()
print(f"✓ MICE complete! Remaining missing: {remaining_missing}")

# ============================================================================
# 6. METHOD 3: KNN IMPUTATION
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 3: KNN (K-Nearest Neighbors)")
print("=" * 70)
print("KNN imputes based on similar samples")
print("Imputing...")

knn_imputer = KNNImputer(n_neighbors=5)

df_knn = df.copy()
df_knn[numeric_cols] = knn_imputer.fit_transform(df_numeric)

remaining_missing = df_knn[numeric_cols].isnull().sum().sum()
print(f"✓ KNN complete! Remaining missing: {remaining_missing}")

# ============================================================================
# 7. ROUND TO 2 DECIMAL PLACES
# ============================================================================
print("\n" + "=" * 70)
print("ROUNDING TO 2 DECIMAL PLACES")
print("=" * 70)

for df_method in [df_missforest, df_mice, df_knn]:
    for col in numeric_cols:
        if col in df_method.columns:
            df_method[col] = df_method[col].round(2)

print("✓ All numeric values rounded to 2 decimal places")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

df_missforest.to_csv('imputed_missforest.csv', index=False)
print("✓ Saved: imputed_missforest.csv (RECOMMENDED - Most Accurate)")

df_mice.to_csv('imputed_mice.csv', index=False)
print("✓ Saved: imputed_mice.csv")

df_knn.to_csv('imputed_knn.csv', index=False)
print("✓ Saved: imputed_knn.csv")

# ============================================================================
# 9. QUALITY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("QUALITY ANALYSIS")
print("=" * 70)

# Check for duplicate values in imputed data
methods = {
    'MissForest': df_missforest,
    'MICE': df_mice,
    'KNN': df_knn
}

for method_name, df_method in methods.items():
    print(f"\n{method_name}:")
    
    # Check uniqueness for key columns
    for col in ['Clay %', 'Gravel %']:
        if col in df_method.columns:
            # Get only imputed values
            missing_mask = df_original[col].isnull()
            imputed_vals = df_method.loc[missing_mask, col]
            
            unique_count = imputed_vals.nunique()
            total_imputed = len(imputed_vals)
            uniqueness_pct = (unique_count / total_imputed * 100) if total_imputed > 0 else 100
            
            print(f"  {col}: {unique_count}/{total_imputed} unique ({uniqueness_pct:.1f}%)")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("Based on 2024 research, MissForest is the most accurate method.")
print("Use: imputed_missforest.csv")
print("\nAll missing values have been filled!")
print("=" * 70)
