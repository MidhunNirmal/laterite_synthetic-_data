"""Debug training script to identify the error."""

import pandas as pd
import numpy as np
import traceback
import os

# Change to script directory
os.chdir(os.path.dirname(__file__))

try:
    print("Loading data...")
    data = pd.read_csv('../imputed_missforest.csv')
    
    # Remove index column if present
    if 'Sl. No' in data.columns:
        data = data.drop(columns=['Sl. No'])
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Define column types
    numerical_columns = [
        'Specific Gravity', 'Gravel %', 'Sand %', 'Silt %', 'Clay %',
        'Liquid Limit %', 'Plastic Limit %', 'Plasticity Index %',
        'OMC %', 'MDD kN/m3', 'CBR % UnSoaked', 'CBR % Soaked', 'wPI'
    ]
    categorical_columns = ['Location', 'Soil Classification']
    
    print("\nLoading mixed KDE encoder...")
    from mixed_kde_encoder import MixedKDEEncoder
    
    print("Creating encoder...")
    kde_encoder = MixedKDEEncoder(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        bandwidth='silverman',
        grid_points=1000,
        tail_clip=0.001
    )
    
    print("Fitting encoder...")
    encoded_data = kde_encoder.fit_transform(data)
    print(f"Encoded shape: {encoded_data.shape}")
    print(f"Encoded range: [{encoded_data.min():.4f}, {encoded_data.max():.4f}]")
    
    print("\nLoading Gaussian copula...")
    from gaussian_copula import GaussianCopula
    
    print("Creating copula...")
    copula = GaussianCopula(regularization=1e-6)
    
    print("Fitting copula...")
    z_data = copula.fit_transform(encoded_data)
    
    print(f"Z-space shape: {z_data.shape}")
    print(f"Z-space range: [{z_data.min():.4f}, {z_data.max():.4f}]")
    print("\n✓ All steps successful!")
    
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}")
    print(f"Message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
