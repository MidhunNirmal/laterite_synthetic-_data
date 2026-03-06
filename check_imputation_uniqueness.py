import pandas as pd
import numpy as np

# Load original and imputed data
df_orig = pd.read_csv(
    'laterite.csv',
    na_values=['_', '', ' ', 'NaN', 'nan', 'NA', 'na', 'N/A', 'n/a'],
    skipinitialspace=True
)
df_orig = df_orig.loc[:, ~df_orig.columns.str.contains('^Unnamed', na=False)]

df_imputed = pd.read_csv('cT_gan.csv')

print("=" * 70)
print("IMPUTATION UNIQUENESS ANALYSIS")
print("=" * 70)

# Check key columns
columns_to_check = ['Clay %', 'Gravel %', 'Sand %', 'Plastic Limit %', 'Plasticity Index %']

for col in columns_to_check:
    if col in df_orig.columns:
        print(f"\n{col}:")
        print("-" * 50)
        
        # Original stats
        missing_in_orig = df_orig[col].isnull().sum()
        non_missing_in_orig = df_orig[col].notnull().sum()
        
        print(f"  Original: {non_missing_in_orig} non-missing, {missing_in_orig} missing")
        
        # Get the imputed values only (where original was NaN)
        imputed_mask = df_orig[col].isnull()
        imputed_values = df_imputed.loc[imputed_mask, col]
        
        print(f"  Imputed values count: {len(imputed_values)}")
        print(f"  Unique imputed values: {imputed_values.nunique()}")
        
        # Check for duplicates IN THE IMPUTED VALUES ONLY
        imputed_duplicates = imputed_values.value_counts()
        imputed_duplicates = imputed_duplicates[imputed_duplicates > 1]
        
        if len(imputed_duplicates) > 0:
            print(f"  ⚠ WARNING: Duplicate values in imputed data:")
            for val, count in imputed_duplicates.items():
                print(f"    Value {val}: appears {count} times")
        else:
            print(f"  ✓ All imputed values are unique!")
        
        # Check total (including original)
        print(f"  Total unique values (original + imputed): {df_imputed[col].nunique()}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total rows: {len(df_imputed)}")
print(f"Missing values after imputation: {df_imputed.isnull().sum().sum()}")
