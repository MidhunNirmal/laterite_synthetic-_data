# Laterite Dataset: Synthetic Data Generation Readiness Report
**Analysis Date:** 2026-01-23
**Dataset:** laterite.csv

---
## 1. Executive Summary

This analysis covers a laterite soil dataset with **53 samples** across **16 features**.

**Key Findings:**

- **Multiple Scenarios:** 16 data sources detected
  - Recommendation: DO NOT POOL - separate by scenario
- **Missing Data:** 10 columns with missing values
- **Small Dataset:** Only 53 samples - limited for GAN training

---
## 2. Dataset Overview

- **Rows:** 53
- **Columns:** 16
- **Numeric Columns:** 13
- **Categorical Columns:** 2
- **Duplicate Rows:** 0

---
## 3. Scenario Consistency Analysis

**Identified Scenarios:** 16

1. MI
2. SM
3. MH
4. SM-SC
5. ML
6. SW
7. SC
8. SW-SM
9. SP-SM
10. SC-GM
11. SM-GC
12. SC - GC
13. SM-CI
14. GM-SC
15. GM-GC
16. GC-SM

**Significant distribution differences:** 49.3%

**Recommendation:** DO NOT POOL - separate by scenario

---
## 4. Statistical Distribution Analysis

### Specific Gravity
- Range: [2.22, 2.80]
- Mean: 2.62, Std Dev: 0.16
- Skewness: -1.00, Kurtosis: -0.23
- Characteristics: highly_skewed

### Gravel %
- Range: [2.00, 55.00]
- Mean: 21.16, Std Dev: 10.83
- Skewness: 0.76, Kurtosis: 1.23

### Sand %
- Range: [21.00, 89.60]
- Mean: 54.70, Std Dev: 15.16
- Skewness: 0.31, Kurtosis: -0.30

### Silt %
- Range: [1.50, 52.00]
- Mean: 16.99, Std Dev: 11.30
- Skewness: 0.90, Kurtosis: 0.39

### Clay %
- Range: [4.00, 43.00]
- Mean: 14.73, Std Dev: 6.85
- Skewness: 2.12, Kurtosis: 6.93
- Characteristics: highly_skewed, heavy_tailed

### Liquid Limit %
- Range: [20.00, 58.80]
- Mean: 36.76, Std Dev: 8.75
- Skewness: 0.51, Kurtosis: 0.11

### Plastic Limit %
- Range: [10.00, 47.20]
- Mean: 22.79, Std Dev: 9.12
- Skewness: 0.64, Kurtosis: -0.37

### Plasticity Index %
- Range: [3.20, 31.70]
- Mean: 15.76, Std Dev: 6.98
- Skewness: 0.23, Kurtosis: -0.78

### OMC %
- Range: [9.90, 36.50]
- Mean: 16.59, Std Dev: 4.29
- Skewness: 2.07, Kurtosis: 8.12
- Characteristics: highly_skewed, heavy_tailed

### MDD kN/m3
- Range: [0.15, 2.39]
- Mean: 1.78, Std Dev: 0.32
- Skewness: -2.46, Kurtosis: 11.88
- Characteristics: highly_skewed, heavy_tailed

**Scale Compatibility:** incompatible
- Scale ratio: 230.66

---
## 5. Missing Data Analysis

| Column | Missing % | Strategy |
|--------|-----------|----------|
| Specific Gravity | 15.1% | IMPUTE_KNN |
| Gravel % | 9.4% | IMPUTE_MEDIAN |
| Clay % | 35.8% | RETAIN_AS_CATEGORY |
| Liquid Limit % | 5.7% | IMPUTE_MEDIAN |
| Plastic Limit % | 5.7% | REPLACE_WITH_ZERO |
| Plasticity Index % | 0.0% | REPLACE_WITH_ZERO |
| OMC % | 11.3% | IMPUTE_KNN |
| MDD kN/m3 | 11.3% | IMPUTE_KNN |
| CBR % Soaked | 49.1% | RETAIN_AS_CATEGORY |
| wPI | 81.1% | DROP_COLUMN |

---
## 6. Outlier Assessment

### Specific Gravity
- Outliers: 2 (4.4%)
- Classification: NATURAL_VARIATION
- Treatment: RETAIN

### Gravel %
- Outliers: 2 (4.2%)
- Classification: NATURAL_VARIATION
- Treatment: RETAIN

### Silt %
- Outliers: 1 (1.9%)
- Classification: NATURAL_VARIATION
- Treatment: RETAIN

### Clay %
- Outliers: 2 (5.9%)
- Classification: NATURAL_VARIATION
- Treatment: RETAIN

### OMC %
- Outliers: 1 (2.1%)
- Classification: NATURAL_VARIATION
- Treatment: RETAIN

### MDD kN/m3
- Outliers: 1 (2.1%)
- Classification: NATURAL_VARIATION
- Treatment: RETAIN

---
## 7. Synthetic Data Generation Readiness

### Recommended Approach

1. **Scenario-Wise Generation:** Generate synthetic data separately for each scenario
2. **Conditional GAN:** Use scenario labels as conditioning variables
3. **Normalization:** StandardScaler required due to scale incompatibility
4. **Small Sample:** Consider TVAE or Gaussian Copula over CTGAN (<100 samples)

---
## 8. Preprocessing Pipeline

1. **Special Values:** Replace 'NP' with 0 for plasticity columns
2. **Column Removal:** Drop columns with >50% missing
3. **Outlier Treatment:** Cap extreme outliers
4. **Imputation:** Median for <10% missing
5. **Normalization:** StandardScaler for all numeric features
6. **Encoding:** Label encoding for soil classifications

**Output:** `laterite_preprocessed.csv`

---
## 9. Risks & Limitations

### Risks
- Small sample size (53 samples) - risk of mode collapse
- Data quality issues - manual verification recommended
- Scenario heterogeneity may introduce artificial variance

### Assumptions
- Missing values assumed MCAR (Missing Completely At Random)
- 'NP' represents zero plasticity index
- Outliers within 3xIQR assumed natural variation

### Limitations
- Cannot definitively determine MCAR vs MAR vs MNAR
- Physical constraints not explicitly enforced
- Some scenario labels incomplete

---
## 10. Recommendations

### Training Parameters
- **Epochs:** 500-1000 (small dataset requires more iterations)
- **Batch Size:** 8-16 (small batches for limited samples)
- **Model:** TVAE preferred over CTGAN for small tabular data

### Validation
1. KS tests for univariate distributions
2. Correlation matrix comparison
3. Physical validity checks (percentages sum to 100%)
4. Atterberg limit relationships

### Alternative Approaches
- Gaussian Copula (better for correlations with limited data)
- KDE-Copula GAN (hybrid approach)
- SMOTE-like methods (targeted augmentation)

---
## Conclusion

The laterite dataset presents challenges due to small sample size, multiple scenarios, and data quality issues. However, with proper preprocessing and scenario-aware modeling, synthetic generation is feasible.

**Recommendation:** Use TVAE or Gaussian Copula over CTGAN due to limited data.
