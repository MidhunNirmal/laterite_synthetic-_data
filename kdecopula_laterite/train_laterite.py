"""
Training Script for KDE-Copula GAN on Laterite Dataset.

This script:
1. Loads pre-imputed laterite data
2. Separates numerical and categorical columns
3. Fits mixed KDE encoder
4. Transforms to copula space
5. Trains WGAN-GP
6. Saves all components
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import yaml
from datetime import datetime

# Import modules
from mixed_kde_encoder import MixedKDEEncoder
from gaussian_copula import GaussianCopula
from trainer import train_wgan_gp


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    print("=" * 70)
    print("KDE-Copula GAN Training for Laterite Dataset")
    print("=" * 70)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    
    # Set random seed
    seed = config.get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"\n[1/6] Loading data...")
    # Load pre-imputed data
    data_path = os.path.join(os.path.dirname(__file__), config['paths']['input_data'])
    data = pd.read_csv(data_path)
    
    # Remove index column if present
    if 'Sl. No' in data.columns:
        data = data.drop(columns=['Sl. No'])
    
    print(f"   Data shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Get column specifications
    numerical_columns = config['data']['numerical_columns']
    categorical_columns = config['data']['categorical_columns']
    
    print(f"\n   Numerical columns: {len(numerical_columns)}")
    print(f"   Categorical columns: {len(categorical_columns)}")
    
    # Verify columns exist
    for col in numerical_columns + categorical_columns:
        if col not in data.columns:
            print(f"   WARNING: Column '{col}' not found in data")
    
    # Filter to only existing columns
    numerical_columns = [col for col in numerical_columns if col in data.columns]
    categorical_columns = [col for col in categorical_columns if col in data.columns]
    
    print(f"\n[2/6] Fitting Mixed KDE Encoder...")
    # Initialize and fit mixed KDE encoder
    kde_encoder = MixedKDEEncoder(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        bandwidth=config['kde']['bandwidth'],
        grid_points=config['kde']['grid_points'],
        tail_clip=config['kde']['tail_clip']
    )
    
    # Fit and transform
    encoded_data = kde_encoder.fit_transform(data)
    print(f"   Encoded shape: {encoded_data.shape}")
    print(f"   Encoded range: [{encoded_data.min():.4f}, {encoded_data.max():.4f}]")
    
    print(f"\n[3/6] Transforming to Gaussian Copula Space...")
    # Fit Gaussian copula
    copula = GaussianCopula(regularization=float(config['copula']['regularization']))
    z_data = copula.fit_transform(encoded_data)
    
    print(f"   Z-space shape: {z_data.shape}")
    print(f"   Z-space range: [{z_data.min():.4f}, {z_data.max():.4f}]")
    print(f"   Correlation matrix condition number: {np.linalg.cond(copula.correlation_matrix):.2f}")
    
    print(f"\n[4/6] Training WGAN-GP...")
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Train GAN
    generator, discriminator, history = train_wgan_gp(
        data=z_data,
        noise_dim=config['noise_dim'],
        hidden_dims=config['generator_dim'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        n_critic=config['discriminator_steps'],
        gp_weight=config['gradient_penalty_weight'],
        device=device,
        verbose=True
    )
    
    print(f"\n[5/6] Saving model...")
    # Prepare model data
    model_data = {
        'kde_encoder': kde_encoder,
        'copula': copula,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'config': config,
        'training_history': history,
        'trained_date': datetime.now().isoformat()
    }
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), config['paths']['output_model'])
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"   Model saved to: {model_path}")
    
    print(f"\n[6/6] Training Summary:")
    print(f"   Final Generator Loss: {history['g_loss'][-1]:.4f}")
    print(f"   Final Discriminator Loss: {history['d_loss'][-1]:.4f}")
    print(f"   Final Wasserstein Distance: {history['wasserstein_distance'][-1]:.4f}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
