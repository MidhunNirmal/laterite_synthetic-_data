"""
Mixed KDE Encoder for KDE-Copula GAN.
Handles mixed data types (numerical + categorical) for tabular data generation.

This module extends the standard KDE encoder to:
1. Apply KDE to numerical columns only
2. Integrate categorical encoder for categorical columns
3. Produce unified percentile representation
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate
from typing import List, Dict, Tuple, Optional
import pickle
from dataclasses import dataclass

from categorical_encoder import CategoricalEncoder


@dataclass
class KDEFeatureInfo:
    """Stores KDE information for a single numerical feature."""
    name: str
    kde: stats.gaussian_kde
    x_min: float
    x_max: float
    x_grid: np.ndarray
    cdf_grid: np.ndarray
    bandwidth: float


class MixedKDEEncoder:
    """
    Mixed KDE encoder for numerical and categorical features.
    
    Applies KDE to numerical features and one-hot encoding to categorical features,
    producing a unified representation suitable for copula transformation.
    """
    
    def __init__(
        self,
        numerical_columns: List[str],
        categorical_columns: List[str],
        bandwidth: str = 'silverman',
        grid_points: int = 1000,
        tail_clip: float = 0.001
    ):
        """
        Initialize mixed KDE encoder.
        
        Args:
            numerical_columns: List of numerical column names
            categorical_columns: List of categorical column names
            bandwidth: Bandwidth selection method for KDE
            grid_points: Number of points for CDF grid
            tail_clip: Clamp percentiles to [clip, 1-clip]
        """
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.bandwidth = bandwidth
        self.grid_points = grid_points
        self.tail_clip = tail_clip
        
        # KDE for numerical features
        self.feature_info: Dict[str, KDEFeatureInfo] = {}
        
        # Categorical encoder
        self.categorical_encoder = CategoricalEncoder()
        
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> 'MixedKDEEncoder':
        """
        Fit KDE for numerical features and encoder for categorical features.
        
        Args:
            data: DataFrame with both numerical and categorical columns
            
        Returns:
            self
        """
        # Fit KDE for numerical columns
        for col in self.numerical_columns:
            if col not in data.columns:
                raise ValueError(f"Numerical column {col} not found in data")
            
            feature_data = data[col].values
            
            # Remove NaN values for fitting
            valid_data = feature_data[~np.isnan(feature_data)]
            
            if len(valid_data) < 2:
                raise ValueError(f"Feature {col} has insufficient valid data")
            
            # Trim outliers for robust KDE (1st-99th percentile)
            p1, p99 = np.percentile(valid_data, [1, 99])
            trimmed_data = valid_data[(valid_data >= p1) & (valid_data <= p99)]
            
            if len(trimmed_data) < 2:
                trimmed_data = valid_data
            
            # Fit Gaussian KDE
            kde = stats.gaussian_kde(trimmed_data, bw_method=self.bandwidth)
            
            # Compute grid for CDF
            x_min, x_max = valid_data.min(), valid_data.max()
            margin = (x_max - x_min) * 0.1
            x_grid = np.linspace(x_min - margin, x_max + margin, self.grid_points)
            
            # Compute CDF at grid points using cumulative integration
            # This is compatible with NumPy 2.x
            pdf_grid = kde.evaluate(x_grid)
            cdf_grid = np.zeros_like(x_grid)
            for i in range(1, len(x_grid)):
                dx = x_grid[i] - x_grid[i-1]
                cdf_grid[i] = cdf_grid[i-1] + 0.5 * (pdf_grid[i] + pdf_grid[i-1]) * dx
            
            # Normalize to [0, 1]
            if cdf_grid[-1] > 0:
                cdf_grid = cdf_grid / cdf_grid[-1]

            
            self.feature_info[col] = KDEFeatureInfo(
                name=col,
                kde=kde,
                x_min=x_min,
                x_max=x_max,
                x_grid=x_grid,
                cdf_grid=cdf_grid,
                bandwidth=kde.factor
            )
        
        # Fit categorical encoder
        if self.categorical_columns:
            self.categorical_encoder.fit(data, self.categorical_columns)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data to unified representation.
        
        For numerical: percentiles via KDE CDF
        For categorical: one-hot encoding
        
        Args:
            data: DataFrame with both numerical and categorical columns
            
        Returns:
            Combined array of shape (n_samples, n_numerical + n_categorical_onehot)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        n_samples = len(data)
        components = []
        
        # Transform numerical columns to percentiles
        if self.numerical_columns:
            percentiles = np.zeros((n_samples, len(self.numerical_columns)))
            
            for i, col in enumerate(self.numerical_columns):
                info = self.feature_info[col]
                feature_data = data[col].values
                
                # Compute percentiles via interpolation
                percentiles[:, i] = np.interp(feature_data, info.x_grid, info.cdf_grid)
                
                # Clamp to avoid numerical issues
                percentiles[:, i] = np.clip(
                    percentiles[:, i],
                    self.tail_clip,
                    1 - self.tail_clip
                )
            
            components.append(percentiles)
        
        # Transform categorical columns to one-hot then to percentiles
        if self.categorical_columns:
            one_hot = self.categorical_encoder.transform(data, self.categorical_columns)
            # Convert one-hot (0/1) to smooth percentiles to avoid copula issues
            # Map 0 -> tail_clip, 1 -> 1-tail_clip
            one_hot_percentiles = one_hot * (1 - 2 * self.tail_clip) + self.tail_clip
            components.append(one_hot_percentiles)
        
        # Combine all components
        return np.hstack(components) if components else np.array([]).reshape(n_samples, 0)
    
    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        """
        Transform unified representation back to original space.
        
        Args:
            encoded: Combined array of percentiles and one-hot encodings
            
        Returns:
            DataFrame with original columns
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before inverse transform")
        
        n_samples = encoded.shape[0]
        result = {}
        
        idx = 0
        
        # Inverse transform numerical columns
        for col in self.numerical_columns:
            info = self.feature_info[col]
            
            # Clamp percentiles
            u_clamped = np.clip(
                encoded[:, idx],
                self.tail_clip,
                1 - self.tail_clip
            )
            
            # Build inverse CDF interpolator
            # Remove duplicate CDF values to avoid interpolation errors
            cdf_unique_mask = np.concatenate([[True], np.diff(info.cdf_grid) > 1e-10])
            cdf_unique = info.cdf_grid[cdf_unique_mask]
            x_unique = info.x_grid[cdf_unique_mask]
            
            # Use linear interpolation if too few unique points for cubic
            interp_kind = 'cubic' if len(cdf_unique) >= 4 else 'linear'
            
            inverse_cdf = interpolate.interp1d(
                cdf_unique,
                x_unique,
                kind=interp_kind,
                bounds_error=False,
                fill_value=(info.x_min, info.x_max)
            )
            
            # Apply inverse CDF
            values = inverse_cdf(u_clamped)
            
            # Enforce bounds
            values = np.clip(values, info.x_min, info.x_max)
            
            # Handle NaN
            nan_mask = np.isnan(values)
            if nan_mask.any():
                values[nan_mask] = (info.x_min + info.x_max) / 2
            
            result[col] = values
            idx += 1
        
        # Inverse transform categorical columns
        if self.categorical_columns:
            n_cat_dims = self.categorical_encoder.get_total_dimensions(self.categorical_columns)
            cat_encoded_percentiles = encoded[:, idx:idx + n_cat_dims]
            
            # Convert smooth percentiles back to one-hot (0/1)
            # Values close to 1-tail_clip become 1, values close to tail_clip become 0
            cat_one_hot = (cat_encoded_percentiles - self.tail_clip) / (1 - 2 * self.tail_clip)
            cat_one_hot = np.clip(cat_one_hot, 0, 1)
            
            cat_df = self.categorical_encoder.inverse_transform(
                cat_one_hot,
                self.categorical_columns,
                use_argmax=True
            )
            result.update(cat_df.to_dict('series'))
        
        return pd.DataFrame(result)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def get_total_dimensions(self) -> int:
        """Get total dimensions after encoding."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted first")
        
        n_numerical = len(self.numerical_columns)
        n_categorical = self.categorical_encoder.get_total_dimensions(self.categorical_columns)
        return n_numerical + n_categorical
    
    def save(self, filepath: str):
        """Save encoder to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted encoder")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'MixedKDEEncoder':
        """Load encoder from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
