"""
Generator Network for KDE-Copula GAN.
Transforms random noise to synthetic copula latent vectors.
"""

import torch
import torch.nn as nn
from typing import List


class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim)
        )
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class Generator(nn.Module):
    """
    Generator network for KDE-Copula GAN.
    
    Input: noise vector ~ N(0, I)
    Output: synthetic copula latent ẑ
    """
    
    def __init__(
        self,
        noise_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256, 256]
    ):
        """
        Initialize generator.
        
        Args:
            noise_dim: Dimension of input noise vector
            output_dim: Dimension of output (number of features)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        
        # Build network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(noise_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        
        # Hidden layers with residual connections
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(ResidualBlock(hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(nn.LeakyReLU(0.2))
        
        # Output layer (no activation - output is in Gaussian space)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic copula latent vectors.
        
        Args:
            noise: Noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Generated data of shape (batch_size, output_dim)
        """
        return self.network(noise)
    
    def sample(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample synthetic data.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        
        noise = torch.randn(n_samples, self.noise_dim, device=device)
        return self.forward(noise)
