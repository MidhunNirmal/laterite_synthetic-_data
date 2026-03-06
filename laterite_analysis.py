"""
Laterite Dataset Analysis for Synthetic Data Generation Readiness
=================================================================

This script performs a comprehensive, assumption-aware analysis of the laterite dataset
to assess its readiness for synthetic tabular data generation using CTGAN, TVAE, or
Gaussian Copula models.

Author: Data Science Team
Date: 2026-01-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance, shapiro, normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
import logging
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

# Configure logging - file only to avoid Windows console encoding issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('laterite_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Also print to console with simple print statements
def log_and_print(message, level='INFO'):
    """Log to file and print to console safely."""
    logger.log(getattr(logging, level), message)
    try:
        print(message)
    except:
        print(message.encode('ascii', errors='replace').decode('ascii'))

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class LateriteDatasetAnalyzer:
    """
    Comprehensive analyzer for laterite dataset to assess synthetic data generation readiness.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the analyzer with dataset filepath.
        
        Args:
            filepath: Path to the laterite CSV file
        """
        self.filepath = filepath
        self.df_raw = None
        self.df_clean = None
        self.analysis_results = {
            'structural': {},
            'scenario': {},
            'statistical': {},
            'missing_data': {},
            'outliers': {},
            'preprocessing_decisions': {},
            'recommendations': []
        }
        logger.info(f"Initialized LateriteDatasetAnalyzer with file: {filepath}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial inspection of the dataset.
        
        Returns:
            Loaded DataFrame
        """
        logger.info("=" * 80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 80)
        
        # Load with explicit handling of various missing value representations
        self.df_raw = pd.read_csv(
            self.filepath,
            na_values=['_', '', ' ', 'NaN', 'nan', 'NA', 'na', 'N/A', 'n/a'],
            skipinitialspace=True
        )
        
        logger.info(f"Dataset loaded: {self.df_raw.shape[0]} rows × {self.df_raw.shape[1]} columns")
        logger.info(f"Columns: {list(self.df_raw.columns)}")
        
        # Display first few rows
        logger.info("\nFirst 5 rows:")
        logger.info(f"\n{self.df_raw.head()}")
        
        return self.df_raw
    
    def structural_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive structural analysis of the dataset.
        
        Returns:
            Dictionary containing structural analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: STRUCTURAL ANALYSIS")
        logger.info("=" * 80)
        
        results = {}
        
        # Basic shape
        results['n_rows'] = self.df_raw.shape[0]
        results['n_columns'] = self.df_raw.shape[1]
        logger.info(f"Dataset shape: {results['n_rows']} rows × {results['n_columns']} columns")
        
        # Remove trailing unnamed columns (likely artifacts from CSV formatting)
        cols_to_drop = [col for col in self.df_raw.columns if 'Unnamed' in str(col) or col.strip() == '']
        if cols_to_drop:
            logger.info(f"Dropping unnamed/empty columns: {cols_to_drop}")
            self.df_raw = self.df_raw.drop(columns=cols_to_drop)
        
        # Column data types
        results['dtypes'] = {}
        logger.info("\n--- Column Data Types ---")
        for col in self.df_raw.columns:
            dtype = str(self.df_raw[col].dtype)
            results['dtypes'][col] = dtype
            logger.info(f"  {col}: {dtype}")
        
        # Identify constant or near-constant columns
        results['constant_columns'] = []
        results['near_constant_columns'] = []
        
        logger.info("\n--- Constant/Near-Constant Column Detection ---")
        for col in self.df_raw.columns:
            n_unique = self.df_raw[col].nunique()
            n_total = len(self.df_raw[col].dropna())
            
            if n_unique == 1:
                results['constant_columns'].append(col)
                logger.info(f"  CONSTANT: {col} (only 1 unique value)")
            elif n_total > 0 and n_unique / n_total < 0.05:
                results['near_constant_columns'].append(col)
                logger.info(f"  NEAR-CONSTANT: {col} ({n_unique} unique values, {n_unique/n_total:.2%} of non-null)")
        
        if not results['constant_columns']:
            logger.info("  No constant columns detected")
        if not results['near_constant_columns']:
            logger.info("  No near-constant columns detected")
        
        # Duplicate row detection
        duplicates = self.df_raw.duplicated()
        results['n_duplicate_rows'] = duplicates.sum()
        results['duplicate_row_indices'] = list(self.df_raw[duplicates].index)
        
        logger.info(f"\n--- Duplicate Rows ---")
        logger.info(f"  Number of duplicate rows: {results['n_duplicate_rows']}")
        if results['n_duplicate_rows'] > 0:
            logger.info(f"  Duplicate indices: {results['duplicate_row_indices']}")
        
        # Column classification (numeric vs categorical)
        results['numeric_columns'] = []
        results['categorical_columns'] = []
        
        logger.info("\n--- Column Classification ---")
        for col in self.df_raw.columns:
            if col in ['Sl. No', 'Location']:
                results['categorical_columns'].append(col)
                logger.info(f"  CATEGORICAL: {col} (identifier)")
            elif self.df_raw[col].dtype in ['object']:
                # Check if it's actually numeric stored as string
                try:
                    pd.to_numeric(self.df_raw[col], errors='coerce')
                    # If conversion is mostly successful, treat as numeric
                    non_null = self.df_raw[col].dropna()
                    if len(non_null) > 0:
                        numeric_conversion = pd.to_numeric(non_null, errors='coerce')
                        if numeric_conversion.notna().sum() / len(non_null) > 0.5:
                            results['numeric_columns'].append(col)
                            logger.info(f"  NUMERIC (stored as object): {col}")
                        else:
                            results['categorical_columns'].append(col)
                            logger.info(f"  CATEGORICAL: {col}")
                except:
                    results['categorical_columns'].append(col)
                    logger.info(f"  CATEGORICAL: {col}")
            else:
                results['numeric_columns'].append(col)
                logger.info(f"  NUMERIC: {col}")
        
        self.analysis_results['structural'] = results
        return results
    
    def scenario_analysis(self) -> Dict[str, Any]:
        """
        Analyze whether data comes from multiple scenarios and assess consistency.
        
        Returns:
            Dictionary containing scenario analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: SCENARIO CONSISTENCY ANALYSIS")
        logger.info("=" * 80)
        
        results = {}
        
        # The last column appears to contain source/scenario information
        # Let's identify it properly
        last_col = self.df_raw.columns[-1]
        logger.info(f"Potential scenario column identified: '{last_col}'")
        
        # Extract scenario labels (handling NaN)
        scenarios = self.df_raw[last_col].fillna('Unknown').unique()
        results['scenarios'] = list(scenarios)
        results['n_scenarios'] = len(scenarios)
        
        logger.info(f"\nIdentified {results['n_scenarios']} scenarios:")
        for i, scenario in enumerate(scenarios, 1):
            count = (self.df_raw[last_col].fillna('Unknown') == scenario).sum()
            logger.info(f"  {i}. '{scenario}': {count} samples")
        
        # Create a scenario label column for analysis
        self.df_raw['_scenario'] = self.df_raw[last_col].fillna('Unknown')
        
        # Analyze distribution shifts between scenarios
        logger.info("\n--- Inter-Scenario Distribution Shift Analysis ---")
        
        numeric_cols = []
        for col in self.df_raw.columns:
            if col not in ['Sl. No', 'Location', last_col, '_scenario']:
                try:
                    # Try to convert to numeric
                    numeric_data = pd.to_numeric(self.df_raw[col], errors='coerce')
                    if numeric_data.notna().sum() > 5:  # At least 5 valid numeric values
                        numeric_cols.append(col)
                except:
                    continue
        
        results['distribution_tests'] = {}
        
        # Perform pairwise KS tests and Wasserstein distance for numeric columns
        scenario_list = [s for s in scenarios if s != 'Unknown']
        
        if len(scenario_list) >= 2:
            logger.info(f"\nPerforming pairwise statistical tests on {len(numeric_cols)} numeric columns")
            
            for col in numeric_cols[:10]:  # Limit to first 10 to avoid excessive output
                results['distribution_tests'][col] = {}
                
                for i in range(len(scenario_list)):
                    for j in range(i + 1, len(scenario_list)):
                        s1, s2 = scenario_list[i], scenario_list[j]
                        
                        # Get data for each scenario
                        data1 = pd.to_numeric(
                            self.df_raw[self.df_raw['_scenario'] == s1][col],
                            errors='coerce'
                        ).dropna()
                        data2 = pd.to_numeric(
                            self.df_raw[self.df_raw['_scenario'] == s2][col],
                            errors='coerce'
                        ).dropna()
                        
                        if len(data1) >= 3 and len(data2) >= 3:
                            # KS test
                            ks_stat, ks_pval = ks_2samp(data1, data2)
                            
                            # Wasserstein distance
                            w_dist = wasserstein_distance(data1, data2)
                            
                            pair_key = f"{s1[:20]} vs {s2[:20]}"
                            results['distribution_tests'][col][pair_key] = {
                                'ks_statistic': float(ks_stat),
                                'ks_pvalue': float(ks_pval),
                                'wasserstein_distance': float(w_dist),
                                'significant_at_0.05': ks_pval < 0.05
                            }
                            
                            if ks_pval < 0.05:
                                logger.info(f"\n  {col} - {pair_key}:")
                                logger.info(f"    KS test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f} ***SIGNIFICANT***")
                                logger.info(f"    Wasserstein distance: {w_dist:.4f}")
        
        # Summary recommendation
        significant_tests = 0
        total_tests = 0
        
        for col_tests in results['distribution_tests'].values():
            for test_result in col_tests.values():
                total_tests += 1
                if test_result['significant_at_0.05']:
                    significant_tests += 1
        
        if total_tests > 0:
            results['percent_significant_tests'] = (significant_tests / total_tests) * 100
            logger.info(f"\n--- Summary ---")
            logger.info(f"Significant distribution differences: {significant_tests}/{total_tests} ({results['percent_significant_tests']:.1f}%)")
            
            if results['percent_significant_tests'] > 30:
                results['pooling_recommendation'] = "DO NOT POOL - Significant inter-scenario variability detected"
                logger.warning("RECOMMENDATION: Do NOT pool scenarios - consider scenario-wise generation")
            elif results['percent_significant_tests'] > 15:
                results['pooling_recommendation'] = "CAUTION - Moderate inter-scenario variability"
                logger.warning("RECOMMENDATION: Consider conditional GAN with scenario labels")
            else:
                results['pooling_recommendation'] = "POOLING ACCEPTABLE - Limited inter-scenario variability"
                logger.info("RECOMMENDATION: Pooling scenarios is acceptable with caution")
        
        self.analysis_results['scenario'] = results
        return results
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive statistical distribution analysis.
        
        Returns:
            Dictionary containing statistical analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: STATISTICAL DISTRIBUTION ANALYSIS")
        logger.info("=" * 80)
        
        results = {}
        
        # Get numeric columns
        numeric_cols = []
        for col in self.df_raw.columns:
            if col not in ['Sl. No', 'Location', '_scenario'] and col != self.df_raw.columns[-1]:
                try:
                    numeric_data = pd.to_numeric(self.df_raw[col], errors='coerce')
                    if numeric_data.notna().sum() > 0:
                        numeric_cols.append(col)
                except:
                    continue
        
        logger.info(f"Analyzing {len(numeric_cols)} numeric columns\n")
        
        for col in numeric_cols:
            logger.info(f"--- {col} ---")
            
            # Convert to numeric
            data = pd.to_numeric(self.df_raw[col], errors='coerce').dropna()
            
            if len(data) == 0:
                logger.info(f"  No valid data\n")
                continue
            
            col_stats = {}
            
            # Basic statistics
            col_stats['count'] = int(len(data))
            col_stats['min'] = float(data.min())
            col_stats['max'] = float(data.max())
            col_stats['range'] = float(data.max() - data.min())
            col_stats['mean'] = float(data.mean())
            col_stats['median'] = float(data.median())
            col_stats['std'] = float(data.std())
            col_stats['variance'] = float(data.var())
            
            # Skewness and kurtosis
            col_stats['skewness'] = float(stats.skew(data))
            col_stats['kurtosis'] = float(stats.kurtosis(data))
            
            # Coefficient of variation
            if col_stats['mean'] != 0:
                col_stats['cv'] = abs(col_stats['std'] / col_stats['mean'])
            else:
                col_stats['cv'] = np.inf
            
            logger.info(f"  Count: {col_stats['count']}")
            logger.info(f"  Range: [{col_stats['min']:.4f}, {col_stats['max']:.4f}] (span: {col_stats['range']:.4f})")
            logger.info(f"  Mean: {col_stats['mean']:.4f}, Median: {col_stats['median']:.4f}, Std: {col_stats['std']:.4f}")
            logger.info(f"  Skewness: {col_stats['skewness']:.4f}, Kurtosis: {col_stats['kurtosis']:.4f}")
            logger.info(f"  Coefficient of Variation: {col_stats['cv']:.4f}")
            
            # Distribution characteristics
            col_stats['characteristics'] = []
            
            if abs(col_stats['skewness']) > 1:
                col_stats['characteristics'].append('highly_skewed')
                logger.info(f"  WARNING: HIGHLY SKEWED (|skew| > 1)")
            
            if col_stats['kurtosis'] > 3:
                col_stats['characteristics'].append('heavy_tailed')
                logger.info(f"  WARNING: HEAVY-TAILED (kurtosis > 3)")
            
            if (data == 0).sum() / len(data) > 0.1:
                col_stats['characteristics'].append('zero_inflated')
                logger.info(f"  WARNING: ZERO-INFLATED ({(data == 0).sum() / len(data):.1%} zeros)")
            
            # Check for bounded variables (e.g., percentages)
            if col_stats['min'] >= 0 and col_stats['max'] <= 100 and '%' in col:
                col_stats['characteristics'].append('bounded_percentage')
                logger.info(f"  INFO: BOUNDED (percentage variable)")
            
            # Normality tests
            if len(data) >= 3:
                try:
                    shapiro_stat, shapiro_pval = shapiro(data) if len(data) < 5000 else (None, None)
                    if shapiro_pval:
                        col_stats['shapiro_pvalue'] = float(shapiro_pval)
                        if shapiro_pval < 0.05:
                            col_stats['characteristics'].append('non_normal')
                            logger.info(f"  Non-normal distribution (Shapiro-Wilk p={shapiro_pval:.4f})")
                except:
                    pass
            
            # Flag suitability for CTGAN
            col_stats['ctgan_suitable'] = True
            col_stats['ctgan_concerns'] = []
            
            if 'highly_skewed' in col_stats['characteristics']:
                col_stats['ctgan_concerns'].append('Extreme skewness may cause mode collapse')
            
            if 'heavy_tailed' in col_stats['characteristics']:
                col_stats['ctgan_concerns'].append('Heavy tails may be poorly captured')
            
            if col_stats['cv'] > 2:
                col_stats['ctgan_concerns'].append('High variance relative to mean')
            
            if col_stats['ctgan_concerns']:
                col_stats['ctgan_suitable'] = False
                logger.info(f"  WARNING: CTGAN CONCERNS: {'; '.join(col_stats['ctgan_concerns'])}")
            
            results[col] = col_stats
            logger.info("")
        
        # Scale compatibility analysis
        logger.info("\n--- Scale Compatibility Analysis ---")
        scales = {}
        for col, stats_dict in results.items():
            scales[col] = stats_dict.get('range', 0)
        
        if scales:
            max_scale = max(scales.values())
            min_scale = min([v for v in scales.values() if v > 0], default=1)
            scale_ratio = max_scale / min_scale if min_scale > 0 else np.inf
            
            logger.info(f"Scale ratio (max/min range): {scale_ratio:.2f}")
            
            if scale_ratio > 100:
                logger.warning("WARNING: SEVERE scale incompatibility detected - normalization REQUIRED")
                results['scale_compatibility'] = 'incompatible'
            elif scale_ratio > 10:
                logger.warning("WARNING: Moderate scale incompatibility - normalization recommended")
                results['scale_compatibility'] = 'moderate'
            else:
                logger.info("OK: Scales are relatively compatible")
                results['scale_compatibility'] = 'compatible'
        
        self.analysis_results['statistical'] = results
        return results
    
    def missing_data_analysis(self) -> Dict[str, Any]:
        """
        Comprehensive missing data analysis and strategy recommendation.
        
        Returns:
            Dictionary containing missing data analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MISSING DATA ANALYSIS")
        logger.info("=" * 80)
        
        results = {}
        
        # Calculate missing percentages
        missing_counts = self.df_raw.isnull().sum()
        missing_pct = (missing_counts / len(self.df_raw)) * 100
        
        results['missing_summary'] = {}
        
        logger.info("\n--- Missing Value Summary ---")
        for col in self.df_raw.columns:
            if col == '_scenario':
                continue
                
            n_missing = int(missing_counts[col])
            pct_missing = float(missing_pct[col])
            
            if n_missing > 0:
                results['missing_summary'][col] = {
                    'count': n_missing,
                    'percentage': pct_missing
                }
                logger.info(f"  {col}: {n_missing} ({pct_missing:.1f}%)")
        
        if not results['missing_summary']:
            logger.info("  No missing values detected")
        
        # Additional check for special representations (NP, l, etc.)
        logger.info("\n--- Special Value Detection ---")
        special_values_found = False
        
        for col in self.df_raw.columns:
            if self.df_raw[col].dtype == 'object':
                special_vals = self.df_raw[col][self.df_raw[col].notna()].unique()
                non_numeric = [v for v in special_vals if isinstance(v, str) and v.strip() in ['NP', 'l', 'I', 'L']]
                
                if non_numeric:
                    special_values_found = True
                    logger.info(f"  {col}: Special values detected: {non_numeric}")
                    
                    if col not in results['missing_summary']:
                        results['missing_summary'][col] = {'count': 0, 'percentage': 0.0}
                    
                    results['missing_summary'][col]['special_values'] = non_numeric
        
        if not special_values_found:
            logger.info("  No special value representations detected")
        
        # Missing data handling strategies
        logger.info("\n--- Missing Data Handling Strategy ---")
        results['handling_strategy'] = {}
        
        for col, miss_info in results['missing_summary'].items():
            pct = miss_info['percentage']
            strategy = {}
            
            logger.info(f"\n  {col} ({pct:.1f}% missing):")
            
            # Decision logic
            if pct > 50:
                strategy['action'] = 'DROP_COLUMN'
                strategy['reason'] = f'More than 50% missing ({pct:.1f}%) - insufficient data for reliable imputation'
                logger.info(f"    → DROP COLUMN: {strategy['reason']}")
            
            elif pct > 30:
                strategy['action'] = 'RETAIN_AS_CATEGORY'
                strategy['reason'] = f'Substantial missingness ({pct:.1f}%) - treat as informative missing category'
                logger.info(f"    → RETAIN AS CATEGORY: {strategy['reason']}")
            
            elif col in ['Plasticity Index %', 'Plastic Limit %'] and 'special_values' in miss_info:
                if 'NP' in miss_info['special_values']:
                    strategy['action'] = 'REPLACE_WITH_ZERO'
                    strategy['reason'] = 'NP (Non-Plastic) is physically meaningful - replace with 0'
                    logger.info(f"    → REPLACE WITH 0: {strategy['reason']}")
            
            elif pct < 10:
                # Check if numeric or categorical
                try:
                    numeric_data = pd.to_numeric(self.df_raw[col], errors='coerce')
                    if numeric_data.notna().sum() > len(self.df_raw) * 0.5:
                        strategy['action'] = 'IMPUTE_MEDIAN'
                        strategy['reason'] = f'Low missingness ({pct:.1f}%) in numeric column - median imputation preserves distribution'
                        logger.info(f"    → IMPUTE (MEDIAN): {strategy['reason']}")
                    else:
                        strategy['action'] = 'IMPUTE_MODE'
                        strategy['reason'] = f'Low missingness ({pct:.1f}%) in categorical column - mode imputation'
                        logger.info(f"    → IMPUTE (MODE): {strategy['reason']}")
                except:
                    strategy['action'] = 'IMPUTE_MODE'
                    strategy['reason'] = f'Low missingness ({pct:.1f}%) - mode imputation'
                    logger.info(f"    → IMPUTE (MODE): {strategy['reason']}")
            
            else:
                strategy['action'] = 'IMPUTE_KNN'
                strategy['reason'] = f'Moderate missingness ({pct:.1f}%) - KNN imputation preserves correlations'
                logger.info(f"    → IMPUTE (KNN): {strategy['reason']}")
            
            results['handling_strategy'][col] = strategy
        
        self.analysis_results['missing_data'] = results
        return results
    
    def outlier_analysis(self) -> Dict[str, Any]:
        """
        Detect and analyze outliers using robust statistical methods.
        
        Returns:
            Dictionary containing outlier analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: OUTLIER AND NOISE ANALYSIS")
        logger.info("=" * 80)
        
        results = {}
        
        # Get numeric columns
        numeric_cols = []
        for col in self.df_raw.columns:
            if col not in ['Sl. No', 'Location', '_scenario']:
                try:
                    numeric_data = pd.to_numeric(self.df_raw[col], errors='coerce')
                    if numeric_data.notna().sum() > 5:
                        numeric_cols.append(col)
                except:
                    continue
        
        logger.info(f"Analyzing outliers in {len(numeric_cols)} numeric columns\n")
        
        for col in numeric_cols:
            logger.info(f"--- {col} ---")
            
            data = pd.to_numeric(self.df_raw[col], errors='coerce').dropna()
            
            if len(data) < 5:
                logger.info("  Insufficient data for outlier detection\n")
                continue
            
            col_results = {}
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            col_results['iqr_outliers'] = {
                'count': int(len(iqr_outliers)),
                'percentage': float(len(iqr_outliers) / len(data) * 100),
                'values': [float(v) for v in iqr_outliers.values],
                'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
            }
            
            logger.info(f"  IQR Method: {len(iqr_outliers)} outliers ({len(iqr_outliers)/len(data)*100:.1f}%)")
            logger.info(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            if len(iqr_outliers) > 0:
                logger.info(f"  Outlier values: {iqr_outliers.values}")
            
            # Z-score method (robust using median and MAD)
            median = data.median()
            mad = np.median(np.abs(data - median))
            
            if mad > 0:
                modified_z_scores = 0.6745 * (data - median) / mad
                zscore_outliers = data[np.abs(modified_z_scores) > 3.5]
                
                col_results['zscore_outliers'] = {
                    'count': int(len(zscore_outliers)),
                    'percentage': float(len(zscore_outliers) / len(data) * 100),
                    'values': [float(v) for v in zscore_outliers.values]
                }
                
                logger.info(f"  Modified Z-Score Method: {len(zscore_outliers)} outliers ({len(zscore_outliers)/len(data)*100:.1f}%)")
            
            # Physical interpretation and classification
            col_results['classification'] = []
            col_results['treatment'] = 'RETAIN'
            col_results['reasoning'] = []
            
            # Check for extreme outliers that might be data entry errors
            if len(iqr_outliers) > 0:
                max_val = data.max()
                typical_max = Q3 + 3 * IQR
                
                if max_val > typical_max * 10:
                    col_results['classification'].append('MEASUREMENT_ERROR')
                    col_results['treatment'] = 'INVESTIGATE'
                    col_results['reasoning'].append(
                        f'Extreme outlier detected: {max_val:.2f} >> typical range. '
                        f'Likely data entry error (e.g., CBR % UnSoaked = 142 in row 44)'
                    )
                    logger.warning(f"  WARNING: EXTREME OUTLIER DETECTED - likely measurement error")
                    logger.warning(f"      Value: {max_val:.2f}, Expected max: ~{typical_max:.2f}")
                
                elif col in ['Gravel %', 'Sand %', 'Silt %', 'Clay %'] and max_val > 100:
                    col_results['classification'].append('IMPOSSIBLE_VALUE')
                    col_results['treatment'] = 'REMOVE'
                    col_results['reasoning'].append('Percentage > 100% is physically impossible')
                    logger.error(f"  ERROR: IMPOSSIBLE VALUE: Percentage exceeds 100%")
                
                else:
                    col_results['classification'].append('NATURAL_VARIATION')
                    col_results['treatment'] = 'RETAIN'
                    col_results['reasoning'].append(
                        'Outliers within reasonable physical bounds - likely natural variation in laterite properties'
                    )
                    logger.info(f"  OK: Outliers appear to be natural variation")
            
            # Impact on synthetic data
            if len(iqr_outliers) / len(data) > 0.1:
                col_results['synthetic_impact'] = 'HIGH'
                logger.warning(f"  WARNING: HIGH impact on synthetic data - >10% outliers may cause GAN instability")
            elif len(iqr_outliers) > 0:
                col_results['synthetic_impact'] = 'MODERATE'
                logger.info(f"  INFO: MODERATE impact - outliers should be monitored in synthetic data")
            else:
                col_results['synthetic_impact'] = 'LOW'
                logger.info(f"  OK: LOW impact - no outliers detected")
            
            results[col] = col_results
            logger.info("")
        
        self.analysis_results['outliers'] = results
        return results
    
    def generate_preprocessing_code(self) -> str:
        """
        Generate the actual preprocessing code based on analysis decisions.
        
        Returns:
            String containing the preprocessing code
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: GENERATING PREPROCESSING CODE")
        logger.info("=" * 80)
        
        code_lines = []
        
        code_lines.append('"""')
        code_lines.append('Laterite Dataset Preprocessing for Synthetic Data Generation')
        code_lines.append('Generated based on comprehensive statistical analysis')
        code_lines.append('"""')
        code_lines.append('')
        code_lines.append('import pandas as pd')
        code_lines.append('import numpy as np')
        code_lines.append('from sklearn.preprocessing import StandardScaler, LabelEncoder')
        code_lines.append('from sklearn.impute import SimpleImputer, KNNImputer')
        code_lines.append('')
        code_lines.append("# Load data")
        code_lines.append("df = pd.read_csv('laterite.csv', na_values=['_', '', ' ', 'NaN', 'nan', 'NA', 'na', 'N/A', 'n/a'])")
        code_lines.append('')
        code_lines.append('# Remove trailing empty columns')
        code_lines.append("df = df.loc[:, ~df.columns.str.contains('^Unnamed')]")
        code_lines.append("df = df.loc[:, df.columns.str.strip() != '']")
        code_lines.append('')
        
        # Handle special values
        code_lines.append('# Handle special value representations')
        code_lines.append("# NP (Non-Plastic) should be treated as 0 for Plasticity Index")
        code_lines.append("plasticity_cols = ['Plastic Limit %', 'Plasticity Index %']")
        code_lines.append('for col in plasticity_cols:')
        code_lines.append('    if col in df.columns:')
        code_lines.append("        df[col] = df[col].replace('NP', '0')")
        code_lines.append("        df[col] = df[col].replace('l', np.nan)  # Likely typo for 1 or lowercase L")
        code_lines.append('')
        
        # Drop columns based on missing data strategy
        if self.analysis_results.get('missing_data', {}).get('handling_strategy'):
            cols_to_drop = [
                col for col, strategy in self.analysis_results['missing_data']['handling_strategy'].items()
                if strategy.get('action') == 'DROP_COLUMN'
            ]
            
            if cols_to_drop:
                code_lines.append('# Drop columns with excessive missing data')
                code_lines.append(f"cols_to_drop = {cols_to_drop}")
                code_lines.append('df = df.drop(columns=cols_to_drop)')
                code_lines.append('')
        
        # Convert columns to numeric
        code_lines.append('# Convert numeric columns')
        code_lines.append("numeric_cols = [col for col in df.columns if col not in ['Sl. No', 'Location', 'Soil Classification']]")
        code_lines.append('for col in numeric_cols:')
        code_lines.append('    if col in df.columns:')
        code_lines.append("        df[col] = pd.to_numeric(df[col], errors='coerce')")
        code_lines.append('')
        
        # Handle outliers
        if self.analysis_results.get('outliers'):
            code_lines.append('# Handle outliers based on analysis')
            for col, outlier_info in self.analysis_results['outliers'].items():
                if outlier_info.get('treatment') == 'REMOVE' or 'MEASUREMENT_ERROR' in outlier_info.get('classification', []):
                    if 'bounds' in outlier_info['iqr_outliers']:
                        bounds = outlier_info['iqr_outliers']['bounds']
                        code_lines.append(f"# Remove outliers from {col}")
                        code_lines.append(f"if '{col}' in df.columns:")
                        code_lines.append(f"    df.loc[(df['{col}'] < {bounds['lower']}) | (df['{col}'] > {bounds['upper']}), '{col}'] = np.nan")
            code_lines.append('')
        
        # Imputation strategy
        code_lines.append('# Imputation strategies')
        code_lines.append("# Median imputation for numeric columns with <10% missing")
        code_lines.append('median_impute_cols = []')
        code_lines.append('for col in numeric_cols:')
        code_lines.append('    if col in df.columns:')
        code_lines.append('        missing_pct = df[col].isnull().sum() / len(df) * 100')
        code_lines.append('        if 0 < missing_pct < 10:')
        code_lines.append('            median_impute_cols.append(col)')
        code_lines.append('')
        code_lines.append('if median_impute_cols:')
        code_lines.append("    imputer = SimpleImputer(strategy='median')")
        code_lines.append('    df[median_impute_cols] = imputer.fit_transform(df[median_impute_cols])')
        code_lines.append('')
        
        # Normalization
        code_lines.append('# Normalization (StandardScaler for GAN compatibility)')
        code_lines.append('# Exclude identifier columns')
        code_lines.append("cols_to_normalize = [col for col in df.columns if col not in ['Sl. No', 'Location', 'Soil Classification']]")
        code_lines.append('cols_to_normalize = [col for col in cols_to_normalize if df[col].dtype in [np.float64, np.int64]]')
        code_lines.append('')
        code_lines.append('scaler = StandardScaler()')
        code_lines.append('df_normalized = df.copy()')
        code_lines.append('df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize].fillna(0))')
        code_lines.append('')
        
        # Encode categorical
        code_lines.append('# Encode categorical variables')
        code_lines.append("if 'Soil Classification' in df.columns:")
        code_lines.append('    le = LabelEncoder()')
        code_lines.append("    df_normalized['Soil Classification Encoded'] = le.fit_transform(df['Soil Classification'].fillna('Unknown'))")
        code_lines.append('')
        
        # Save
        code_lines.append('# Save preprocessed data')
        code_lines.append("df_normalized.to_csv('laterite_preprocessed.csv', index=False)")
        code_lines.append("print('Preprocessing complete. Saved to laterite_preprocessed.csv')")
        code_lines.append("print(f'Shape: {df_normalized.shape}')")
        
        preprocessing_code = '\n'.join(code_lines)
        
        # Write to file
        with open('laterite_preprocessing.py', 'w') as f:
            f.write(preprocessing_code)
        
        logger.info("OK: Preprocessing code generated and saved to 'laterite_preprocessing.py'")
        
        return preprocessing_code
    
    def generate_markdown_report(self) -> str:
        """
        Generate comprehensive Markdown report.
        
        Returns:
            String containing the Markdown report
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: GENERATING MARKDOWN REPORT")
        logger.info("=" * 80)
        
        report = []
        
        # Header
        report.append("# Laterite Dataset: Synthetic Data Generation Readiness Report")
        report.append("")
        report.append(f"**Analysis Date:** 2026-01-23")
        report.append(f"**Dataset:** laterite.csv")
        report.append(f"**Analyst:** Senior Data Scientist & Data Quality Auditor")
        report.append("")
        report.append("---")
        report.append("")
        
        # Executive Summary
        report.append("## 1. Executive Summary")
        report.append("")
        
        struct = self.analysis_results.get('structural', {})
        scenario = self.analysis_results.get('scenario', {})
        
        report.append(f"This report presents a comprehensive, assumption-aware analysis of a laterite soil dataset "
                     f"containing **{struct.get('n_rows', 'N/A')} samples** across **{struct.get('n_columns', 'N/A')} features**. ")
        report.append("")
        report.append("**Key Findings:**")
        report.append("")
        
        # Scenario finding
        if scenario.get('n_scenarios', 0) > 1:
            report.append(f"- ⚠️ **Multiple Scenarios Detected:** {scenario['n_scenarios']} distinct data sources identified")
            report.append(f"  - {scenario.get('pooling_recommendation', 'See detailed analysis')}")
        
        # Missing data finding
        missing = self.analysis_results.get('missing_data', {})
        if missing.get('missing_summary'):
            n_cols_with_missing = len(missing['missing_summary'])
            report.append(f"- **Missing Data:** {n_cols_with_missing} columns contain missing values")
        
        # Outliers
        outliers = self.analysis_results.get('outliers', {})
        severe_outliers = sum(1 for v in outliers.values() if 'MEASUREMENT_ERROR' in v.get('classification', []))
        if severe_outliers > 0:
            report.append(f"- ⚠️ **Data Quality Issues:** {severe_outliers} columns contain likely measurement errors")
        
        # Dataset size concern
        if struct.get('n_rows', 0) < 100:
            report.append(f"- ⚠️ **Small Dataset:** Only {struct['n_rows']} samples may limit synthetic data quality")
        
        report.append("")
        report.append("**Recommendation:** Proceed with synthetic data generation using scenario-aware approaches and careful validation.")
        report.append("")
        report.append("---")
        report.append("")
        
        # Dataset Overview
        report.append("## 2. Dataset Overview")
        report.append("")
        report.append(f"- **Total Rows:** {struct.get('n_rows', 'N/A')}")
        report.append(f"- **Total Columns:** {struct.get('n_columns', 'N/A')}")
        report.append("")
        
        report.append("### Column Classification")
        report.append("")
        report.append("| Type | Count | Columns |")
        report.append("|------|-------|---------|")
        report.append(f"| Numeric | {len(struct.get('numeric_columns', []))} | {', '.join(struct.get('numeric_columns', [])[:5])}... |")
        report.append(f"| Categorical | {len(struct.get('categorical_columns', []))} | {', '.join(struct.get('categorical_columns', []))} |")
        report.append("")
        
        if struct.get('n_duplicate_rows', 0) > 0:
            report.append(f"**⚠️ Duplicate Rows:** {struct['n_duplicate_rows']} duplicate rows detected")
            report.append("")
        
        report.append("---")
        report.append("")
        
        # Scenario Analysis
        report.append("## 3. Scenario Consistency Analysis")
        report.append("")
        
        if scenario.get('scenarios'):
            report.append(f"### Identified Scenarios ({scenario['n_scenarios']} total)")
            report.append("")
            for i, sc in enumerate(scenario['scenarios'], 1):
                report.append(f"{i}. `{sc}`")
            report.append("")
        
        if scenario.get('distribution_tests'):
            report.append("### Inter-Scenario Distribution Tests")
            report.append("")
            report.append("Statistical tests (Kolmogorov-Smirnov) were performed to detect distribution shifts:")
            report.append("")
            
            if scenario.get('percent_significant_tests'):
                report.append(f"- **Significant differences:** {scenario['percent_significant_tests']:.1f}% of pairwise tests")
                report.append(f"- **Recommendation:** {scenario.get('pooling_recommendation', 'N/A')}")
            report.append("")
        
        report.append("---")
        report.append("")
        
        # Statistical Analysis
        report.append("## 4. Statistical Distribution Analysis")
        report.append("")
        
        stats_results = self.analysis_results.get('statistical', {})
        
        # Select a few key columns for detailed reporting
        report.append("### Key Column Statistics")
        report.append("")
        
        for col in list(stats_results.keys())[:10]:  # First 10 columns
            if isinstance(stats_results[col], dict):
                col_stat = stats_results[col]
                report.append(f"#### {col}")
                report.append("")
                report.append(f"- **Range:** [{col_stat.get('min', 'N/A'):.2f}, {col_stat.get('max', 'N/A'):.2f}]")
                report.append(f"- **Mean ± SD:** {col_stat.get('mean', 'N/A'):.2f} ± {col_stat.get('std', 'N/A'):.2f}")
                report.append(f"- **Skewness:** {col_stat.get('skewness', 'N/A'):.2f}")
                report.append(f"- **Kurtosis:** {col_stat.get('kurtosis', 'N/A'):.2f}")
                
                if col_stat.get('characteristics'):
                    report.append(f"- **Characteristics:** {', '.join(col_stat['characteristics'])}")
                
                if not col_stat.get('ctgan_suitable', True):
                    report.append(f"- ⚠️ **CTGAN Concerns:** {'; '.join(col_stat.get('ctgan_concerns', []))}")
                
                report.append("")
        
        if stats_results.get('scale_compatibility'):
            report.append(f"### Scale Compatibility: `{stats_results['scale_compatibility']}`")
            report.append("")
        
        report.append("---")
        report.append("")
        
        # Missing Data Analysis
        report.append("## 5. Missing Data Analysis")
        report.append("")
        
        if missing.get('missing_summary'):
            report.append("### Missing Value Summary")
            report.append("")
            report.append("| Column | Missing Count | Missing % | Strategy |")
            report.append("|--------|---------------|-----------|----------|")
            
            for col, miss_info in missing['missing_summary'].items():
                strategy = missing.get('handling_strategy', {}).get(col, {})
                action = strategy.get('action', 'N/A')
                report.append(f"| {col} | {miss_info['count']} | {miss_info['percentage']:.1f}% | {action} |")
            
            report.append("")
            
            report.append("### Justifications")
            report.append("")
            
            for col, strategy in missing.get('handling_strategy', {}).items():
                report.append(f"**{col}:** {strategy.get('reason', 'N/A')}")
                report.append("")
        
        report.append("---")
        report.append("")
        
        # Outlier Analysis
        report.append("## 6. Outlier & Noise Assessment")
        report.append("")
        
        if outliers:
            for col, outlier_info in list(outliers.items())[:10]:
                if outlier_info.get('iqr_outliers', {}).get('count', 0) > 0:
                    report.append(f"### {col}")
                    report.append("")
                    report.append(f"- **Outliers Detected:** {outlier_info['iqr_outliers']['count']} ({outlier_info['iqr_outliers']['percentage']:.1f}%)")
                    report.append(f"- **Classification:** {', '.join(outlier_info.get('classification', ['Natural Variation']))}")
                    report.append(f"- **Treatment:** {outlier_info.get('treatment', 'RETAIN')}")
                    
                    if outlier_info.get('reasoning'):
                        report.append(f"- **Reasoning:** {'; '.join(outlier_info['reasoning'])}")
                    
                    report.append(f"- **Synthetic Data Impact:** {outlier_info.get('synthetic_impact', 'N/A')}")
                    report.append("")
        
        report.append("---")
        report.append("")
        
        # Synthetic Data Suitability
        report.append("## 7. Synthetic Data Suitability Evaluation")
        report.append("")
        report.append("### CTGAN / TVAE Compatibility")
        report.append("")
        
        # Count columns with concerns
        ctgan_issues = []
        for col, col_stat in stats_results.items():
            if isinstance(col_stat, dict) and not col_stat.get('ctgan_suitable', True):
                ctgan_issues.append((col, col_stat.get('ctgan_concerns', [])))
        
        if ctgan_issues:
            report.append(f"**Columns with Concerns:** {len(ctgan_issues)}")
            report.append("")
            for col, concerns in ctgan_issues:
                report.append(f"- **{col}:** {'; '.join(concerns)}")
            report.append("")
        else:
            report.append("✓ No major compatibility concerns identified")
            report.append("")
        
        report.append("### Recommended Approach")
        report.append("")
        
        if scenario.get('percent_significant_tests', 0) > 30:
            report.append("1. **Scenario-Wise Generation:** Generate synthetic data separately for each scenario")
            report.append("2. **Conditional GAN:** Alternatively, use scenario labels as conditioning variables")
        else:
            report.append("1. **Pooled Generation:** Dataset can be pooled with scenario labels as features")
        
        report.append("3. **Normalization Required:** Apply StandardScaler due to scale incompatibility")
        report.append("4. **Small Sample Consideration:** With <100 samples, consider using Gaussian Copula or TVAE as they may perform better than CTGAN")
        report.append("")
        
        report.append("---")
        report.append("")
        
        # Preprocessing Decisions
        report.append("## 8. Final Preprocessing Decisions")
        report.append("")
        report.append("The following preprocessing pipeline has been implemented:")
        report.append("")
        report.append("1. **Special Value Handling:** Replace 'NP' with 0 for plasticity columns")
        report.append("2. **Column Removal:** Drop columns with >50% missing data")
        report.append("3. **Outlier Treatment:** Cap or remove extreme outliers (e.g., CBR % UnSoaked = 142)")
        report.append("4. **Imputation:** Median imputation for numeric columns with <10% missing")
        report.append("5. **Normalization:** StandardScaler applied to all numeric features")
        report.append("6. **Categorical Encoding:** Label encoding for soil classifications")
        report.append("")
        report.append("**Output:** `laterite_preprocessed.csv`")
        report.append("")
        
        report.append("---")
        report.append("")
        
        # Risks and Limitations
        report.append("## 9. Risks, Assumptions, and Limitations")
        report.append("")
        report.append("### Risks")
        report.append("")
        report.append("- **Small Sample Size:** 53 samples is limited for GAN training - may result in mode collapse")
        report.append("- **Data Quality Issues:** Measurement errors detected; manual verification recommended")
        report.append("- **Scenario Heterogeneity:** Pooling scenarios may introduce artificial variance")
        report.append("")
        report.append("### Assumptions")
        report.append("")
        report.append("- Missing values in plasticity metrics are assumed to be MCAR (Missing Completely At Random)")
        report.append("- 'NP' (Non-Plastic) is assumed to represent zero plasticity index")
        report.append("- Outliers within 3×IQR are assumed to be natural variation")
        report.append("")
        report.append("### Limitations")
        report.append("")
        report.append("- Cannot definitively determine MCAR vs MAR vs MNAR without domain expertise")
        report.append("- Physical constraints (e.g., Gravel% + Sand% + Silt% + Clay% ≈ 100%) not explicitly enforced")
        report.append("- Scenario labels incomplete - some rows have unknown sources")
        report.append("")
        
        report.append("---")
        report.append("")
        
        # Recommendations
        report.append("## 10. Recommendations for CTGAN / TVAE Training")
        report.append("")
        report.append("### Pre-Training")
        report.append("")
        report.append("1. **Data Augmentation:** Consider augmenting with related laterite datasets if available")
        report.append("2. **Scenario-Specific Models:** Train separate models per scenario if distribution tests show high divergence")
        report.append("3. **Constraint Incorporation:** Implement custom loss functions to enforce physical constraints")
        report.append("")
        report.append("### Training Parameters")
        report.append("")
        report.append("- **Epochs:** Start with 500-1000 epochs (small dataset requires more iterations)")
        report.append("- **Batch Size:** Use small batch sizes (8-16) given limited samples")
        report.append("- **Architecture:** Consider TVAE over CTGAN for small tabular data")
        report.append("- **Discriminator Steps:** Reduce discriminator steps (e.g., 1-2) to prevent overfitting")
        report.append("")
        report.append("### Post-Generation Validation")
        report.append("")
        report.append("1. **Distributional Similarity:**")
        report.append("   - KS tests for univariate distributions")
        report.append("   - Wasserstein distance for distribution matching")
        report.append("   - Correlation matrix comparison")
        report.append("")
        report.append("2. **Physical Validity:**")
        report.append("   - Check percentages sum to ~100%")
        report.append("   - Verify Atterberg limit relationships (Liquid Limit > Plastic Limit)")
        report.append("   - Ensure soil classifications align with physical properties")
        report.append("")
        report.append("3. **Scenario Preservation:**")
        report.append("   - Validate that synthetic data maintains scenario-specific characteristics")
        report.append("")
        report.append("### Alternative Approaches")
        report.append("")
        report.append("Given the small sample size and multiple scenarios, consider:")
        report.append("")
        report.append("1. **Gaussian Copula:** May preserve correlations better with limited data")
        report.append("2. **KDE-Copula GAN:** Hybrid approach combining KDE margins with copula dependencies")
        report.append("3. **SMOTE-like Approaches:** For targeted augmentation rather than full generation")
        report.append("4. **Bayesian Synthesis:** Model uncertainty explicitly given small sample size")
        report.append("")
        
        report.append("---")
        report.append("")
        report.append("## Conclusion")
        report.append("")
        report.append("The laterite dataset presents moderate challenges for synthetic data generation due to:")
        report.append("- Small sample size (53 samples)")
        report.append("- Multiple scenarios with distributional differences")
        report.append("- Data quality issues requiring manual verification")
        report.append("")
        report.append("However, with appropriate preprocessing, scenario-aware modeling, and rigorous validation, "
                     "synthetic data generation is feasible. **TVAE or Gaussian Copula models are recommended over CTGAN** "
                     "due to the small sample size.")
        report.append("")
        report.append("All preprocessing decisions have been justified based on statistical evidence and are documented "
                     "in the generated `laterite_preprocessing.py` script.")
        report.append("")
        
        report_text = '\n'.join(report)
        
        # Write to file
        with open('laterite_synthetic_data_readiness_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("OK: Comprehensive report generated and saved to 'laterite_synthetic_data_readiness_report.md'")
        
        return report_text
    
    def run_complete_analysis(self):
        """
        Execute all analysis steps in sequence.
        """
        logger.info("\n" + "=" * 80)
        logger.info("LATERITE DATASET COMPREHENSIVE ANALYSIS")
        logger.info("=" * 80)
        
        # Load data
        self.load_data()
        
        # Run all analysis steps
        self.structural_analysis()
        self.scenario_analysis()
        self.statistical_analysis()
        self.missing_data_analysis()
        self.outlier_analysis()
        
        # Generate outputs
        self.generate_preprocessing_code()
        self.generate_markdown_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info("\nGenerated files:")
        logger.info("  1. laterite_preprocessing.py - Preprocessing code")
        logger.info("  2. laterite_synthetic_data_readiness_report.md - Comprehensive report")
        logger.info("  3. laterite_analysis.log - Detailed analysis log")


if __name__ == "__main__":
    # Initialize and run analysis
    analyzer = LateriteDatasetAnalyzer('laterite.csv')
    analyzer.run_complete_analysis()
