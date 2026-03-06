import pandas as pd
import numpy as np

# Load data
df_orig = pd.read_csv('laterite.csv', na_values=['_', '', ' ', 'NaN', 'nan', 'NA', 'na', 'N/A', 'n/a'])
df_orig = df_orig.loc[:, ~df_orig.columns.str.contains('^Unnamed', na=False)]
df_imputed = pd.read_csv('cT_gan.csv')

print("=" * 70)
print("DETAILED IMPUTATION DUPLICATE ANALYSIS")
print("=" * 70)

columns_to_check = ['Clay %', 'Gravel %', 'Sand %', 'Plastic Limit %', 'Plasticity Index %']

for col in columns_to_check:
    if col in df_orig.columns:
        print(f"\n{col}:")
        print("-" * 50)
        
        # Get mask of which rows were imputed
        missing_mask = df_orig[col].isnull()
        imputed_values = df_imputed.loc[missing_mask, col]
        
        print(f"  Total imputed: {len(imputed_values)}")
        print(f"  Unique imputed: {imputed_values.nunique()}")
        
        # Check for duplicates
        value_counts = imputed_values.value_counts()
        duplicates = value_counts[value_counts > 1]
        
        if len(duplicates) > 0:
            print(f"  ⚠ DUPLICATES FOUND IN IMPUTED VALUES:")
            for val, count in duplicates.items():
                print(f"    Value {val}: appears {count} times")
                # Show which rows got this duplicate value
                dup_rows = df_imputed[missing_mask & (df_imputed[col] == val)]['Sl. No'].values
                print(f"      Rows: {dup_rows}")
        else:
            print(f"  ✓ All imputed values are UNIQUE!")

print("\n" + "=" * 70)
