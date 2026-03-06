"""
Laterite Dataset Preprocessing for Synthetic Data Generation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('laterite.csv', na_values=['_', '', ' ', 'NaN', 'nan', 'NA', 'na', 'N/A', 'n/a'])

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
df = df.loc[:, df.columns.str.strip() != '']

# Handle special values (NP = Non-Plastic)
plasticity_cols = ['Plastic Limit %', 'Plasticity Index %']
for col in plasticity_cols:
    if col in df.columns:
        df[col] = df[col].replace('NP', '0')
        df[col] = df[col].replace('l', np.nan)

# Convert numeric columns
numeric_cols = [col for col in df.columns if col not in ['Sl. No', 'Location', 'Soil Classification']]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle extreme outliers (e.g., CBR % UnSoaked = 142)
if 'CBR % UnSoaked' in df.columns:
    # Cap at 99th percentile
    cap_value = df['CBR % UnSoaked'].quantile(0.99)
    df.loc[df['CBR % UnSoaked'] > cap_value, 'CBR % UnSoaked'] = np.nan

# Median imputation for columns with <10% missing
median_impute_cols = []
for col in numeric_cols:
    if col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if 0 < missing_pct < 10:
            median_impute_cols.append(col)

if median_impute_cols:
    imputer = SimpleImputer(strategy='median')
    df[median_impute_cols] = imputer.fit_transform(df[median_impute_cols])

# Normalization
cols_to_normalize = [col for col in df.columns if col not in ['Sl. No', 'Location', 'Soil Classification']]
cols_to_normalize = [col for col in cols_to_normalize if df[col].dtype in [np.float64, np.int64]]

scaler = StandardScaler()
df_normalized = df.copy()
if cols_to_normalize:
    df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize].fillna(0))

# Encode categorical
if 'Soil Classification' in df.columns:
    le = LabelEncoder()
    df_normalized['Soil Classification Encoded'] = le.fit_transform(df['Soil Classification'].fillna('Unknown'))

# Save
df_normalized.to_csv('laterite_preprocessed.csv', index=False)
print(f'Preprocessing complete. Shape: {df_normalized.shape}')
print('Saved to: laterite_preprocessed.csv')
