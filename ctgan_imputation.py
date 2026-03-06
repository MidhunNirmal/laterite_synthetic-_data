"""
CTGAN-based Missing Data Imputation for Laterite Dataset
This script uses CTGAN to learn the data distribution and impute missing values.
"""

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("CTGAN-based Missing Data Imputation")
print("=" * 70)

# Load the dataset
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

# Preprocess the data
print("\n2. Preprocessing data...")
df = df_original.copy()

# Handle special values for plasticity columns
plasticity_cols = ['Plastic Limit %', 'Plasticity Index %']
for col in plasticity_cols:
    if col in df.columns:
        # Replace 'NP' (Non-Plastic) with 0
        df[col] = df[col].replace('NP', '0')
        df[col] = df[col].replace('l', np.nan)  # Replace invalid 'l' with NaN

# Convert numeric columns to proper numeric type
numeric_cols = [col for col in df.columns if col not in ['Sl. No', 'Location', 'Soil Classification']]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle extreme outliers (cap at 99th percentile to avoid training issues)
if 'CBR % UnSoaked' in df.columns:
    cap_value = df['CBR % UnSoaked'].quantile(0.99)
    df.loc[df['CBR % UnSoaked'] > cap_value, 'CBR % UnSoaked'] = np.nan

print(f"Preprocessed dataset shape: {df.shape}")

# Prepare data for CTGAN
print("\n3. Preparing data for CTGAN...")

# Create a working dataset with only complete rows for CTGAN training
# CTGAN will learn from complete data and then we'll use it to impute
df_complete = df.dropna()
df_with_missing = df[df.isnull().any(axis=1)]

print(f"Complete rows: {len(df_complete)}")
print(f"Rows with missing values: {len(df_with_missing)}")

if len(df_complete) < 10:
    print("\nWARNING: Too few complete rows for reliable CTGAN training.")
    print("Using simple imputation as fallback...")
    # Fallback to median imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
else:
    # Create metadata for CTGAN
    print("\n4. Creating metadata...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_complete)
    
    # Set categorical columns
    if 'Location' in df_complete.columns:
        metadata.update_column('Location', sdtype='categorical')
    if 'Soil Classification' in df_complete.columns:
        metadata.update_column('Soil Classification', sdtype='categorical')
    
    print("Metadata created successfully")
    
    # Train CTGAN on complete data
    print("\n5. Training CTGAN model...")
    print("This may take a few minutes...")
    
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=300,  # Increased epochs for better learning
        batch_size=min(16, len(df_complete)),  # Adjust batch size based on data
        verbose=True
    )
    
    synthesizer.fit(df_complete)
    print("CTGAN training complete!")
    
    # Impute missing values
    print("\n6. Imputing missing values using CTGAN...")
    print("Generating fresh synthetic samples for each row to ensure uniqueness...")
    
    df_imputed = df.copy()
    
    if len(df_with_missing) > 0:
        # Generate a VERY large pool of diverse synthetic samples
        # We need enough diversity to avoid duplicates
        n_synthetic = max(5000, len(df_with_missing) * 100)
        print(f"Generating {n_synthetic} diverse synthetic samples...")
        synthetic_pool = synthesizer.sample(n_synthetic)
        
        # Track which specific values have been assigned to avoid exact duplicates
        assigned_values = {col: set() for col in numeric_cols}
        
        total_rows = len(df_with_missing)
        
        for imputed_count, idx in enumerate(df_with_missing.index, 1):
            if imputed_count % 10 == 0 or imputed_count == total_rows:
                print(f"  Processing row {imputed_count}/{total_rows}...")
            
            row = df.loc[idx]
            missing_cols_in_row = row[row.isnull()].index.tolist()
            known_cols_in_row = row[row.notnull()].index.tolist()
            
            # Filter known columns to only numeric ones for distance calculation
            known_numeric = [c for c in known_cols_in_row if c in numeric_cols]
            
            if len(known_numeric) > 0:
                # Calculate distance for each synthetic sample in the pool
                distances = []
                for _, syn_row in synthetic_pool.iterrows():
                    dist = 0
                    for col in known_numeric:
                        if pd.notnull(row[col]) and pd.notnull(syn_row[col]):
                            dist += (row[col] - syn_row[col]) ** 2
                    distances.append(np.sqrt(dist))
                
                # For each missing column, find a unique value
                for col in missing_cols_in_row:
                    if col in synthetic_pool.columns:
                        selected_val = None
                        
                        # Try progressively larger pools of candidates
                        for top_k in [15, 30, 50, 100, 200, len(distances)]:
                            top_k = min(top_k, len(distances))
                            top_indices = np.argsort(distances)[:top_k]
                            candidates = synthetic_pool.iloc[top_indices][col].values
                            
                            # Find candidates that haven't been used yet
                            unused_candidates = [c for c in candidates if c not in assigned_values[col]]
                            
                            if len(unused_candidates) > 0:
                                # Use row index to seed randomness for this specific choice
                                # This ensures different rows make different choices
                                rng = np.random.RandomState(seed=int(idx) * 1000 + ord(col[0]))
                                selected_val = rng.choice(unused_candidates)
                                break
                        
                        # If we couldn't find an unused value, force uniqueness
                        if selected_val is None:
                            print(f"    WARNING: Pool exhausted for {col}, generating more samples...")
                            # Generate fresh samples until we get a unique one
                            max_new_attempts = 20
                            for attempt in range(max_new_attempts):
                                new_samples = synthesizer.sample(100)
                                for _, new_row in new_samples.iterrows():
                                    candidate = new_row[col]
                                    if candidate not in assigned_values[col]:
                                        selected_val = candidate
                                        break
                                if selected_val is not None:
                                    break
                            
                            # Absolute last resort: add tiny noise to best match
                            if selected_val is None:
                                best_idx = np.argmin(distances)
                                base_val = synthetic_pool.iloc[best_idx][col]
                                # Add tiny random noise to make it unique
                                noise = np.random.uniform(-0.01, 0.01)
                                selected_val = base_val + noise
                                print(f"    RARE: Added noise to create unique value for {col}")
                        
                        # Assign the value and track it
                        df_imputed.loc[idx, col] = selected_val
                        assigned_values[col].add(selected_val)
            else:
                # If no known numeric values, use diverse samples from the pool
                for col in missing_cols_in_row:
                    if col in synthetic_pool.columns:
                        # Try to find an unused value
                        for i in range(len(synthetic_pool)):
                            candidate = synthetic_pool.iloc[i][col]
                            if candidate not in assigned_values[col]:
                                df_imputed.loc[idx, col] = candidate
                                assigned_values[col].add(candidate)
                                break
        
        print("Imputation complete!")

# Validate imputation
print("\n7. Validating imputed data...")
remaining_missing = df_imputed.isnull().sum().sum()
print(f"Remaining missing values: {remaining_missing}")



# Save the imputed dataset
print("\n8. Saving imputed dataset...")
df_imputed.to_csv('cT_gan.csv', index=False)
print("Saved to: cT_gan.csv")

# Summary statistics
print("\n" + "=" * 70)
print("IMPUTATION SUMMARY")
print("=" * 70)
print(f"Original dataset shape: {df_original.shape}")
print(f"Imputed dataset shape: {df_imputed.shape}")
print(f"Total missing values before: {df_original.isnull().sum().sum()}")
print(f"Total missing values after: {df_imputed.isnull().sum().sum()}")

print("\n" + "=" * 70)
print("SUCCESS! Imputed dataset saved as cT_gan.csv")
print("=" * 70)
