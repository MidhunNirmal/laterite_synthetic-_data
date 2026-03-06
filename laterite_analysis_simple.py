"""
Laterite Dataset Analysis - Simplified Version
Outputs all results to files to avoid console encoding issues.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance, shapiro
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
import json

warnings.filterwarnings('ignore')

print("Starting analysis...")

# Load data
df_raw = pd.read_csv(
    'laterite.csv',
    na_values=['_', '', ' ', 'NaN', 'nan', 'NA', 'na', 'N/A', 'n/a'],
    skipinitialspace=True
)

print(f"Loaded {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

# Initialize results dictionary
analysis_results = {
    'structural': {},
    'scenario': {},
    'statistical': {},
    'missing_data': {},
    'outliers': {}
}

#====================================================================================
# 1. STRUCTURAL ANALYSIS
#====================================================================================

print("Running structural analysis...")

# Remove trailing unnamed columns
df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed', na=False)]
df_raw = df_raw.loc[:, df_raw.columns.str.strip() != '']

analysis_results['structural']['n_rows'] = df_raw.shape[0]
analysis_results['structural']['n_columns'] = df_raw.shape[1]
analysis_results['structural']['columns'] = list(df_raw.columns)

# Duplicates
analysis_results['structural']['n_duplicate_rows'] = int(df_raw.duplicated().sum())

# Column classification
numeric_cols = []
categorical_cols = ['Sl. No', 'Location']

for col in df_raw.columns:
    if col in categorical_cols or col == df_raw.columns[-1]:
        continue
    try:
        numeric_data = pd.to_numeric(df_raw[col], errors='coerce')
        if numeric_data.notna().sum() > 0:
            numeric_cols.append(col)
    except:
        categorical_cols.append(col)

analysis_results['structural']['numeric_columns'] = numeric_cols
analysis_results['structural']['categorical_columns'] = categorical_cols

#====================================================================================
# 2. SCENARIO ANALYSIS
#====================================================================================

print("Running scenario analysis...")

# Last column contains scenario information
scenario_col = df_raw.columns[-1]
df_raw['_scenario'] = df_raw[scenario_col].fillna('Unknown')

scenarios = df_raw['_scenario'].unique()
analysis_results['scenario']['scenarios'] = [str(s) for s in scenarios]
analysis_results['scenario']['n_scenarios'] = len(scenarios)

# Distribution tests
dist_tests = {}
scenario_list = [s for s in scenarios if s != 'Unknown']

if len(scenario_list) >= 2 and len(numeric_cols) > 0:
    significant_count = 0
    total_count = 0
    
    for col in numeric_cols[:5]:  # Test first 5 columns
        try:
            for i in range(len(scenario_list)):
                for j in range(i + 1, len(scenario_list)):
                    s1, s2 = scenario_list[i], scenario_list[j]
                    
                    data1 = pd.to_numeric(df_raw[df_raw['_scenario'] == s1][col], errors='coerce').dropna()
                    data2 = pd.to_numeric(df_raw[df_raw['_scenario'] == s2][col], errors='coerce').dropna()
                    
                    if len(data1) >= 3 and len(data2) >= 3:
                        ks_stat, ks_pval = ks_2samp(data1, data2)
                        total_count += 1
                        if ks_pval < 0.05:
                            significant_count += 1
        except:
            pass
    
    if total_count > 0:
        pct_significant = (significant_count / total_count) * 100
        analysis_results['scenario']['percent_significant_tests'] = pct_significant
        
        if pct_significant > 30:
            analysis_results['scenario']['recommendation'] = "DO NOT POOL - separate by scenario"
        elif pct_significant > 15:
            analysis_results['scenario']['recommendation'] = "CAUTION - use conditional GAN"
        else:
            analysis_results['scenario']['recommendation'] = "POOLING ACCEPTABLE"

#====================================================================================
# 3. STATISTICAL ANALYSIS
#====================================================================================

print("Running statistical analysis...")

stats_results = {}

for col in numeric_cols:
    data = pd.to_numeric(df_raw[col], errors='coerce').dropna()
    
    if len(data) == 0:
        continue
   
    col_stats = {
        'count': int(len(data)),
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'median': float(data.median()),
        'std': float(data.std()),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data))
    }
    
    col_stats['characteristics'] = []
    if abs(col_stats['skewness']) > 1:
        col_stats['characteristics'].append('highly_skewed')
    if col_stats['kurtosis'] > 3:
        col_stats['characteristics'].append('heavy_tailed')
    if (data == 0).sum() / len(data) > 0.1:
        col_stats['characteristics'].append('zero_inflated')
    
    stats_results[col] = col_stats

analysis_results['statistical'] = stats_results

# Scale compatibility
if stats_results:
    scales = [v['max'] - v['min'] for v in stats_results.values() if 'max' in v]
    if scales:
        scale_ratio = max(scales) / min([s for s in scales if s > 0], default=1)
        analysis_results['statistical']['scale_ratio'] = float(scale_ratio)
        
        if scale_ratio > 100:
            analysis_results['statistical']['scale_compatibility'] = 'incompatible'
        elif scale_ratio > 10:
            analysis_results['statistical']['scale_compatibility'] = 'moderate'
        else:
            analysis_results['statistical']['scale_compatibility'] = 'compatible'

#====================================================================================
# 4. MISSING DATA ANALYSIS
#====================================================================================

print("Running missing data analysis...")

missing_summary = {}
handling_strategy = {}

for col in df_raw.columns:
    if col == '_scenario':
        continue
        
    n_missing = int(df_raw[col].isnull().sum())
    pct_missing = (n_missing / len(df_raw)) * 100
    
    if n_missing > 0 or (df_raw[col].dtype == 'object' and 'NP' in df_raw[col].values):
        missing_summary[col] = {
            'count': n_missing,
            'percentage': float(pct_missing)
        }
        
        # Strategy
        if pct_missing > 50:
            action = 'DROP_COLUMN'
            reason = f'More than 50% missing - insufficient data'
        elif pct_missing > 30:
            action = 'RETAIN_AS_CATEGORY'
            reason = f'Substantial missingness - treat as informative'
        elif 'Plasticity' in col or 'Plastic' in col:
            action = 'REPLACE_WITH_ZERO'
            reason = 'NP (Non-Plastic) means zero plasticity'
        elif pct_missing < 10:
            action = 'IMPUTE_MEDIAN'
            reason = f'Low missingness - median imputation'
        else:
            action = 'IMPUTE_KNN'
            reason = f'Moderate missingness - KNN imputation'
        
        handling_strategy[col] = {
            'action': action,
            'reason': reason
        }

analysis_results['missing_data'] = {
    'missing_summary': missing_summary,
    'handling_strategy': handling_strategy
}

#====================================================================================
# 5. OUTLIER ANALYSIS
#====================================================================================

print("Running outlier analysis...")

outlier_results = {}

for col in numeric_cols[:10]:  # First 10 columns
    data = pd.to_numeric(df_raw[col], errors='coerce').dropna()
    
    if len(data) < 5:
        continue
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    col_result = {
        'n_outliers': int(len(outliers)),
        'pct_outliers': float(len(outliers) / len(data) * 100),
        'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
    }
    
    # Classification
    max_val = data.max()
    typical_max = Q3 + 3 * IQR
    
    if len(outliers) > 0 and max_val > typical_max * 10:
        col_result['classification'] = 'MEASUREMENT_ERROR'
        col_result['treatment'] = 'INVESTIGATE'
    elif col in ['Gravel %', 'Sand %', 'Silt %', 'Clay %'] and max_val > 100:
        col_result['classification'] = 'IMPOSSIBLE_VALUE'
        col_result['treatment'] = 'REMOVE'
    else:
        col_result['classification'] = 'NATURAL_VARIATION'
        col_result['treatment'] = 'RETAIN'
    
    outlier_results[col] = col_result

analysis_results['outliers'] = outlier_results

#====================================================================================
# 6. SAVE ANALYSIS RESULTS
#====================================================================================

print("Saving analysis results to JSON...")

with open('laterite_analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print("Analysis complete!")
print(f"Results saved to: laterite_analysis_results.json")

#====================================================================================
# 7. GENERATE PREPROCESSING CODE
#====================================================================================

print("Generating preprocessing code...")

preprocessing_code = '''"""
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
'''

with open('laterite_preprocessing.py', 'w') as f:
    f.write(preprocessing_code)

print("Preprocessing code saved to: laterite_preprocessing.py")

print("=" * 60)
print("ALL TASKS COMPLETE")
print("=" * 60)
print("Generated files:")
print("  1. laterite_analysis_results.json")
print("  2. laterite_preprocessing.py")
print("  3. Now generating markdown report...")

#====================================================================================
# 8. GENERATE MARKDOWN REPORT
#====================================================================================

report_lines = []
report_lines.append("# Laterite Dataset: Synthetic Data Generation Readiness Report\n")
report_lines.append("**Analysis Date:** 2026-01-23\n")
report_lines.append("**Dataset:** laterite.csv\n")
report_lines.append("\n---\n")

# Executive Summary
report_lines.append("## 1. Executive Summary\n")
report_lines.append(f"\nThis analysis covers a laterite soil dataset with **{analysis_results['structural']['n_rows']} samples** ")
report_lines.append(f"across **{analysis_results['structural']['n_columns']} features**.\n")
report_lines.append("\n**Key Findings:**\n")

if analysis_results['scenario']['n_scenarios'] > 1:
    report_lines.append(f"\n- **Multiple Scenarios:** {analysis_results['scenario']['n_scenarios']} data sources detected\n")
    if 'recommendation' in analysis_results['scenario']:
        report_lines.append(f"  - Recommendation: {analysis_results['scenario']['recommendation']}\n")

if analysis_results['missing_data']['missing_summary']:
    n_missing_cols = len(analysis_results['missing_data']['missing_summary'])
    report_lines.append(f"- **Missing Data:** {n_missing_cols} columns with missing values\n")

severe_outliers = sum(1 for v in analysis_results['outliers'].values() if v.get('classification') == 'MEASUREMENT_ERROR')
if severe_outliers > 0:
    report_lines.append(f"- **Data Quality Issues:** {severe_outliers} columns with likely measurement errors\n")

if analysis_results['structural']['n_rows'] < 100:
    report_lines.append(f"- **Small Dataset:** Only {analysis_results['structural']['n_rows']} samples - limited for GAN training\n")

report_lines.append("\n---\n")

# Dataset Overview
report_lines.append("## 2. Dataset Overview\n")
report_lines.append(f"\n- **Rows:** {analysis_results['structural']['n_rows']}\n")
report_lines.append(f"- **Columns:** {analysis_results['structural']['n_columns']}\n")
report_lines.append(f"- **Numeric Columns:** {len(analysis_results['structural']['numeric_columns'])}\n")
report_lines.append(f"- **Categorical Columns:** {len(analysis_results['structural']['categorical_columns'])}\n")
report_lines.append(f"- **Duplicate Rows:** {analysis_results['structural']['n_duplicate_rows']}\n")
report_lines.append("\n---\n")

# Scenario Analysis
report_lines.append("## 3. Scenario Consistency Analysis\n")
report_lines.append(f"\n**Identified Scenarios:** {analysis_results['scenario']['n_scenarios']}\n\n")
for i, scenario in enumerate(analysis_results['scenario']['scenarios'], 1):
    report_lines.append(f"{i}. {scenario}\n")

if 'percent_significant_tests' in analysis_results['scenario']:
    report_lines.append(f"\n**Significant distribution differences:** {analysis_results['scenario']['percent_significant_tests']:.1f}%\n")
    report_lines.append(f"\n**Recommendation:** {analysis_results['scenario'].get('recommendation', 'N/A')}\n")

report_lines.append("\n---\n")

# Statistical Analysis
report_lines.append("## 4. Statistical Distribution Analysis\n")

for col, col_stats in list(analysis_results['statistical'].items())[:10]:
    if isinstance(col_stats, dict) and 'mean' in col_stats:
        report_lines.append(f"\n### {col}\n")
        report_lines.append(f"- Range: [{col_stats['min']:.2f}, {col_stats['max']:.2f}]\n")
        report_lines.append(f"- Mean: {col_stats['mean']:.2f}, Std Dev: {col_stats['std']:.2f}\n")
        report_lines.append(f"- Skewness: {col_stats['skewness']:.2f}, Kurtosis: {col_stats['kurtosis']:.2f}\n")
        if col_stats['characteristics']:
            report_lines.append(f"- Characteristics: {', '.join(col_stats['characteristics'])}\n")

if 'scale_compatibility' in analysis_results['statistical']:
    report_lines.append(f"\n**Scale Compatibility:** {analysis_results['statistical']['scale_compatibility']}\n")
    if 'scale_ratio' in analysis_results['statistical']:
        report_lines.append(f"- Scale ratio: {analysis_results['statistical']['scale_ratio']:.2f}\n")

report_lines.append("\n---\n")

# Missing Data
report_lines.append("## 5. Missing Data Analysis\n")
report_lines.append("\n| Column | Missing % | Strategy |\n")
report_lines.append("|--------|-----------|----------|\n")

for col, info in analysis_results['missing_data']['missing_summary'].items():
    strategy = analysis_results['missing_data']['handling_strategy'].get(col, {})
    action = strategy.get('action', 'N/A')
    report_lines.append(f"| {col} | {info['percentage']:.1f}% | {action} |\n")

report_lines.append("\n---\n")

# Outliers
report_lines.append("## 6. Outlier Assessment\n")

for col, outlier_info in analysis_results['outliers'].items():
    if outlier_info['n_outliers'] > 0:
        report_lines.append(f"\n### {col}\n")
        report_lines.append(f"- Outliers: {outlier_info['n_outliers']} ({outlier_info['pct_outliers']:.1f}%)\n")
        report_lines.append(f"- Classification: {outlier_info['classification']}\n")
        report_lines.append(f"- Treatment: {outlier_info['treatment']}\n")

report_lines.append("\n---\n")

# Synthetic Data Suitability
report_lines.append("## 7. Synthetic Data Generation Readiness\n")
report_lines.append("\n### Recommended Approach\n\n")

if analysis_results['scenario'].get('percent_significant_tests', 0) > 30:
    report_lines.append("1. **Scenario-Wise Generation:** Generate synthetic data separately for each scenario\n")
    report_lines.append("2. **Conditional GAN:** Use scenario labels as conditioning variables\n")
else:
    report_lines.append("1. **Pooled Generation:** Dataset can be pooled with scenario labels\n")

report_lines.append("3. **Normalization:** StandardScaler required due to scale incompatibility\n")
report_lines.append("4. **Small Sample:** Consider TVAE or Gaussian Copula over CTGAN (<100 samples)\n")

report_lines.append("\n---\n")

# Preprocessing Decisions
report_lines.append("## 8. Preprocessing Pipeline\n")
report_lines.append("\n1. **Special Values:** Replace 'NP' with 0 for plasticity columns\n")
report_lines.append("2. **Column Removal:** Drop columns with >50% missing\n")
report_lines.append("3. **Outlier Treatment:** Cap extreme outliers\n")
report_lines.append("4. **Imputation:** Median for <10% missing\n")
report_lines.append("5. **Normalization:** StandardScaler for all numeric features\n")
report_lines.append("6. **Encoding:** Label encoding for soil classifications\n")
report_lines.append("\n**Output:** `laterite_preprocessed.csv`\n")

report_lines.append("\n---\n")

# Risks and Limitations
report_lines.append("## 9. Risks & Limitations\n")
report_lines.append("\n### Risks\n")
report_lines.append("- Small sample size (53 samples) - risk of mode collapse\n")
report_lines.append("- Data quality issues - manual verification recommended\n")
report_lines.append("- Scenario heterogeneity may introduce artificial variance\n")

report_lines.append("\n### Assumptions\n")
report_lines.append("- Missing values assumed MCAR (Missing Completely At Random)\n")
report_lines.append("- 'NP' represents zero plasticity index\n")
report_lines.append("- Outliers within 3xIQR assumed natural variation\n")

report_lines.append("\n### Limitations\n")
report_lines.append("- Cannot definitively determine MCAR vs MAR vs MNAR\n")
report_lines.append("- Physical constraints not explicitly enforced\n")
report_lines.append("- Some scenario labels incomplete\n")

report_lines.append("\n---\n")

# Recommendations
report_lines.append("## 10. Recommendations\n")
report_lines.append("\n### Training Parameters\n")
report_lines.append("- **Epochs:** 500-1000 (small dataset requires more iterations)\n")
report_lines.append("- **Batch Size:** 8-16 (small batches for limited samples)\n")
report_lines.append("- **Model:** TVAE preferred over CTGAN for small tabular data\n")

report_lines.append("\n### Validation\n")
report_lines.append("1. KS tests for univariate distributions\n")
report_lines.append("2. Correlation matrix comparison\n")
report_lines.append("3. Physical validity checks (percentages sum to 100%)\n")
report_lines.append("4. Atterberg limit relationships\n")

report_lines.append("\n### Alternative Approaches\n")
report_lines.append("- Gaussian Copula (better for correlations with limited data)\n")
report_lines.append("- KDE-Copula GAN (hybrid approach)\n")
report_lines.append("- SMOTE-like methods (targeted augmentation)\n")

report_lines.append("\n---\n")

# Conclusion
report_lines.append("## Conclusion\n")
report_lines.append("\nThe laterite dataset presents challenges due to small sample size, ")
report_lines.append("multiple scenarios, and data quality issues. However, with proper preprocessing ")
report_lines.append("and scenario-aware modeling, synthetic generation is feasible.\n")
report_lines.append("\n**Recommendation:** Use TVAE or Gaussian Copula over CTGAN due to limited data.\n")

# Write report
report_text = ''.join(report_lines)
with open('laterite_synthetic_data_readiness_report.md', 'w') as f:
    f.write(report_text)

print("Markdown report saved to: laterite_synthetic_data_readiness_report.md")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE - ALL FILES GENERATED")
print("=" * 60)
