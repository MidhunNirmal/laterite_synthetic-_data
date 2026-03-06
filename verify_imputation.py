import pandas as pd
import numpy as np

# Load original and imputed data
df_orig = pd.read_csv('laterite.csv', na_values=['_', '', ' ', 'NaN', 'nan', 'NA', 'na', 'N/A', 'n/a'])
df_orig = df_orig.loc[:, ~df_orig.columns.str.contains('^Unnamed', na=False)]

df_missforest = pd.read_csv('imputed_missforest.csv')
df_mice = pd.read_csv('imputed_mice.csv')
df_knn = pd.read_csv('imputed_knn.csv')

print("=" * 70)
print("DETAILED QUALITY VERIFICATION")
print("=" * 70)

methods = {
    'MissForest (RECOMMENDED)': df_missforest,
    'MICE': df_mice,
    'KNN': df_knn
}

for method_name, df_method in methods.items():
    print(f"\n{method_name}")
    print("-" * 70)
    
    # Overall stats
    print(f"  Total rows: {len(df_method)}")
    print(f"  Missing values: {df_method.isnull().sum().sum()}")
    
    # Check uniqueness for important columns
    cols_to_check = ['Clay %', 'Gravel %', 'Sand %', 'Plastic Limit %']
    
    for col in cols_to_check:
        if col in df_method.columns:
            # Get only imputed values
            missing_mask = df_orig[col].isnull()
            imputed_vals = df_method.loc[missing_mask, col]
            
            if len(imputed_vals) > 0:
                unique_count = imputed_vals.nunique()
                total_imputed = len(imputed_vals)
                uniqueness_pct = (unique_count / total_imputed * 100)
                
                # Check for duplicates
                vc = imputed_vals.value_counts()
                dups = vc[vc > 1]
                
                status = "✓ ALL UNIQUE" if len(dups) == 0 else f"⚠ {len(dups)} duplicates"
                
                print(f"    {col}: {unique_count}/{total_imputed} unique ({uniqueness_pct:.1f}%) - {status}")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("Use 'imputed_missforest.csv' - Research-proven most accurate method")
print("=" * 70)
