"""
WGAN-GP Trainer for KDE-Copula GAN.
Implements Wasserstein GAN with Gradient Penalty training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator network
        real_data: Real data tensor
        fake_data: Generated data tensor
        device: Device to compute on
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    
    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1, device=device)
    
    # Interpolate between real and fake data
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    # Get discriminator output
    d_interpolates = discriminator(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def train_wgan_gp(
    data: np.ndarray,
    noise_dim: int = 128,
    hidden_dims: List[int] = [256, 256, 256],
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 0.0002,
    n_critic: int = 5,
    gp_weight: float = 10.0,
    device: torch.device = None,
    verbose: bool = True
) -> Tuple[Generator, Discriminator, Dict]:
    """
    Train WGAN-GP on copula-transformed data.
    
    Args:
        data: Training data array of shape (n_samples, n_features)
        noise_dim: Dimension of noise vector
        hidden_dims: Hidden layer dimensions
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        n_critic: Number of discriminator steps per generator step
        gp_weight: Gradient penalty weight
        device: Device to train on
        verbose: Whether to show progress
        
    Returns:
        Tuple of (generator, discriminator, training_history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_features = data.shape[1]
    
    # Initialize networks
    generator = Generator(noise_dim, n_features, hidden_dims).to(device)
    discriminator = Discriminator(n_features, hidden_dims).to(device)
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    
    # Convert data to tensor
    data_tensor = torch.FloatTensor(data).to(device)
    
    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'wasserstein_distance': []
    }
    
    # Training loop
    n_batches = len(data) // batch_size
    
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training WGAN-GP")
    
    for epoch in iterator:
        g_losses = []
        d_losses = []
        w_distances = []
        
        # Shuffle data
        perm = torch.randperm(len(data_tensor))
        data_tensor = data_tensor[perm]
        
        for i in range(n_batches):
            # Get real data batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            real_data = data_tensor[start_idx:end_idx]
            
            current_batch_size = real_data.size(0)
            
            # Train discriminator
            for _ in range(n_critic):
                optimizer_d.zero_grad()
                
                # Generate fake data
                noise = torch.randn(current_batch_size, noise_dim, device=device)
                fake_data = generator(noise).detach()
                
                # Discriminator outputs
                d_real = discriminator(real_data)
                d_fake = discriminator(fake_data)
                
                # Wasserstein distance
                wasserstein_distance = d_real.mean() - d_fake.mean()
                
                # Gradient penalty
                gp = compute_gradient_penalty(discriminator, real_data, fake_data, device)
                
                # Discriminator loss
                d_loss = -wasserstein_distance + gp_weight * gp
                
                d_loss.backward()
                optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            fake_data = generator(noise)
            
            g_output = discriminator(fake_data)
            g_loss = -g_output.mean()
            
            g_loss.backward()
            optimizer_g.step()
            
            # Record losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            w_distances.append(wasserstein_distance.item())
        
        # Average losses for epoch
        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        avg_w_distance = np.mean(w_distances)
        
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['wasserstein_distance'].append(avg_w_distance)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - G Loss: {avg_g_loss:.4f}, "
                  f"D Loss: {avg_d_loss:.4f}, W-Distance: {avg_w_distance:.4f}")
    
    return generator, discriminator, history
