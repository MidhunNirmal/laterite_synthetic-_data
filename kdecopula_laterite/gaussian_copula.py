"""
Gaussian Copula Module for KDE-Copula GAN.
Implements copula transformation for mixed-type data.
"""

import numpy as np
from scipy import stats
import pickle
from typing import Optional


class GaussianCopula:
    """
    Gaussian Copula for modeling inter-feature dependencies.
    
    Transforms uniform marginals to Gaussian space where dependencies
    are modeled via the correlation matrix.
    """
    
    def __init__(self, regularization: float = 1e-6):
        """
        Initialize Gaussian copula.
        
        Args:
            regularization: Ridge regularization for correlation matrix
        """
        self.regularization = regularization
        self.correlation_matrix: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, percentiles: np.ndarray) -> 'GaussianCopula':
        """
        Fit Gaussian copula by estimating correlation matrix.
        
        Args:
            percentiles: Array of shape (n_samples, n_features) in (0, 1)
            
        Returns:
            self
        """
        # Transform to Gaussian space: z = Φ⁻¹(u)
        z = stats.norm.ppf(percentiles)
        
        # Handle infinities from extreme percentiles
        z = np.clip(z, -10, 10)
        
        # Estimate correlation matrix
        self.correlation_matrix = np.corrcoef(z.T)
        
        # Validate positive-definiteness
        if not self._is_positive_definite(self.correlation_matrix):
            self.correlation_matrix = self._regularize(self.correlation_matrix)
        
        self.is_fitted = True
        return self
    
    def _is_positive_definite(self, matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def _regularize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Regularize matrix to ensure positive-definiteness.
        Ridge regularization: Σ_reg = Σ + λI
        """
        n = matrix.shape[0]
        reg_matrix = matrix + self.regularization * np.eye(n)
        
        # If still not PD, use eigenvalue correction
        if not self._is_positive_definite(reg_matrix):
            eigenvalues, eigenvectors = np.linalg.eigh(reg_matrix)
            eigenvalues = np.maximum(eigenvalues, self.regularization)
            reg_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return reg_matrix
    
    def transform(self, percentiles: np.ndarray) -> np.ndarray:
        """
        Transform percentiles to Gaussian copula space.
        
        Args:
            percentiles: Array in (0, 1)
            
        Returns:
            Transformed data in Gaussian space
        """
        if not self.is_fitted:
            raise ValueError("Copula must be fitted before transform")
        
        # z = Φ⁻¹(u)
        z = stats.norm.ppf(percentiles)
        z = np.clip(z, -10, 10)
        
        return z
    
    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """
        Transform from Gaussian space back to percentiles.
        
        Args:
            z: Array in Gaussian space
            
        Returns:
            Percentiles in (0, 1)
        """
        # u = Φ(z)
        percentiles = stats.norm.cdf(z)
        percentiles = np.clip(percentiles, 1e-6, 1 - 1e-6)
        
        return percentiles
    
    def fit_transform(self, percentiles: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(percentiles)
        return self.transform(percentiles)
    
    def save(self, filepath: str):
        """Save fitted copula to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted copula")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'GaussianCopula':
        """Load fitted copula from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
