"""
Generation Script for KDE-Copula GAN on Laterite Dataset.

This script:
1. Loads trained model
2. Generates noise → Generator → Z-space
3. Inverse copula transform → percentiles
4. Inverse mixed KDE → original space
5. Saves synthetic data
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch

from mixed_kde_encoder import MixedKDEEncoder
from gaussian_copula import GaussianCopula
from generator import Generator


def generate_synthetic_data(
    model_path: str,
    n_samples: int,
    output_path: str,
    seed: int = 42
):
    """
    Generate synthetic laterite data.
    
    Args:
        model_path: Path to trained model
        n_samples: Number of samples to generate
        output_path: Path to save synthetic data
        seed: Random seed
    """
    print("=" * 70)
    print("KDE-Copula GAN Generation for Laterite Dataset")
    print("=" * 70)
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n[1/5] Loading trained model...")
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    kde_encoder = model_data['kde_encoder']
    copula = model_data['copula']
    config = model_data['config']
    numerical_columns = model_data['numerical_columns']
    categorical_columns = model_data['categorical_columns']
    
    print(f"   Model trained on: {model_data.get('trained_date', 'Unknown')}")
    print(f"   Total features: {len(numerical_columns) + len(categorical_columns)}")
    
    print(f"\n[2/5] Reconstructing generator...")
    # Reconstruct generator
    n_features = kde_encoder.get_total_dimensions()
    generator = Generator(
        noise_dim=config['noise_dim'],
        output_dim=n_features,
        hidden_dims=config['generator_dim']
    )
    generator.load_state_dict(model_data['generator_state_dict'])
    generator.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    print(f"   Using device: {device}")
    
    print(f"\n[3/5] Generating {n_samples} samples...")
    # Generate noise and produce Z-space samples
    with torch.no_grad():
        noise = torch.randn(n_samples, config['noise_dim'], device=device)
        z_synthetic = generator(noise).cpu().numpy()
    
    print(f"   Z-space range: [{z_synthetic.min():.4f}, {z_synthetic.max():.4f}]")
    
    print(f"\n[4/5] Inverse transformations...")
    # Inverse copula transform
    percentiles_synthetic = copula.inverse_transform(z_synthetic)
    print(f"   Percentiles range: [{percentiles_synthetic.min():.4f}, {percentiles_synthetic.max():.4f}]")
    
    # Inverse mixed KDE transform
    synthetic_df = kde_encoder.inverse_transform(percentiles_synthetic)
    
    print(f"   Synthetic data shape: {synthetic_df.shape}")
    
    # Verify column order
    all_columns = numerical_columns + categorical_columns
    synthetic_df = synthetic_df[all_columns]
    
    print(f"\n[5/5] Saving synthetic data...")
    # Save to CSV
    synthetic_df.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    
    # Display sample statistics
    print(f"\n   Sample Statistics:")
    print(f"   Unique Locations: {synthetic_df['Location'].nunique()}")
    print(f"   Unique Soil Classifications: {synthetic_df['Soil Classification'].nunique()}")
    
    for col in numerical_columns[:3]:  # Show first 3 numerical columns
        print(f"   {col}: mean={synthetic_df[col].mean():.2f}, std={synthetic_df[col].std():.2f}")
    
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(description='Generate synthetic laterite data')
    parser.add_argument('--model', type=str, default='laterite_kdecopula_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='laterite_synthetic.csv',
                        help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, args.model)
    output_path = os.path.join(script_dir, args.output)
    
    generate_synthetic_data(
        model_path=model_path,
        n_samples=args.samples,
        output_path=output_path,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
