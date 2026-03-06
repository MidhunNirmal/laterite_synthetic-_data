"""
Categorical Encoder for KDE-Copula GAN.
Handles encoding and decoding of categorical variables for mixed-type tabular data.

This module provides:
1. One-hot encoding for categorical features during training
2. Probabilistic sampling during generation (softmax over logits)
3. Category preservation with original label mapping
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle


class CategoricalEncoder:
    """
    Encoder/decoder for categorical variables in mixed-type data.
    
    Uses one-hot encoding during training and probabilistic decoding
    during generation to preserve categorical distributions.
    """
    
    def __init__(self):
        """Initialize the categorical encoder."""
        self.column_encodings: Dict[str, Dict[str, int]] = {}
        self.column_categories: Dict[str, List[str]] = {}
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame, categorical_columns: List[str]) -> 'CategoricalEncoder':
        """
        Fit encoder on categorical columns.
        
        Args:
            data: DataFrame containing categorical columns
            categorical_columns: List of categorical column names
            
        Returns:
            self
        """
        for col in categorical_columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")
            
            # Get unique categories and create mapping
            categories = sorted(data[col].unique())
            self.column_categories[col] = categories
            self.column_encodings[col] = {cat: idx for idx, cat in enumerate(categories)}
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, categorical_columns: List[str]) -> np.ndarray:
        """
        Transform categorical columns to one-hot encoding.
        
        Args:
            data: DataFrame containing categorical columns
            categorical_columns: List of categorical column names
            
        Returns:
            One-hot encoded array of shape (n_samples, total_one_hot_dims)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        encoded_arrays = []
        
        for col in categorical_columns:
            if col not in self.column_encodings:
                raise ValueError(f"Column {col} was not fitted")
            
            n_samples = len(data)
            n_categories = len(self.column_categories[col])
            
            # Create one-hot encoding
            one_hot = np.zeros((n_samples, n_categories))
            
            for i, val in enumerate(data[col]):
                if val in self.column_encodings[col]:
                    idx = self.column_encodings[col][val]
                    one_hot[i, idx] = 1.0
                else:
                    # Handle unseen category - use most frequent (first category)
                    one_hot[i, 0] = 1.0
            
            encoded_arrays.append(one_hot)
        
        # Concatenate all one-hot encodings
        if encoded_arrays:
            return np.hstack(encoded_arrays)
        else:
            return np.array([]).reshape(len(data), 0)
    
    def inverse_transform(
        self, 
        encoded: np.ndarray, 
        categorical_columns: List[str],
        use_argmax: bool = True
    ) -> pd.DataFrame:
        """
        Transform one-hot encoded data back to categorical values.
        
        Args:
            encoded: One-hot encoded array
            categorical_columns: List of categorical column names
            use_argmax: If True, use argmax. If False, use probabilistic sampling
            
        Returns:
            DataFrame with categorical columns
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before inverse transform")
        
        n_samples = encoded.shape[0]
        result = {}
        
        idx = 0
        for col in categorical_columns:
            if col not in self.column_categories:
                raise ValueError(f"Column {col} was not fitted")
            
            n_categories = len(self.column_categories[col])
            col_encoded = encoded[:, idx:idx + n_categories]
            
            if use_argmax:
                # Use argmax to select category
                category_indices = np.argmax(col_encoded, axis=1)
            else:
                # Use probabilistic sampling
                # Apply softmax to convert to probabilities
                exp_vals = np.exp(col_encoded - np.max(col_encoded, axis=1, keepdims=True))
                probabilities = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
                
                # Sample from the distribution
                category_indices = np.array([
                    np.random.choice(n_categories, p=prob)
                    for prob in probabilities
                ])
            
            # Map indices back to categories
            categories = self.column_categories[col]
            result[col] = [categories[i] for i in category_indices]
            
            idx += n_categories
        
        return pd.DataFrame(result)
    
    def fit_transform(self, data: pd.DataFrame, categorical_columns: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data, categorical_columns)
        return self.transform(data, categorical_columns)
    
    def get_total_dimensions(self, categorical_columns: List[str]) -> int:
        """
        Get total one-hot dimensions for specified columns.
        
        Args:
            categorical_columns: List of categorical column names
            
        Returns:
            Total number of one-hot dimensions
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted first")
        
        total = 0
        for col in categorical_columns:
            if col in self.column_categories:
                total += len(self.column_categories[col])
        return total
    
    def save(self, filepath: str):
        """Save encoder to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted encoder")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'CategoricalEncoder':
        """Load encoder from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
