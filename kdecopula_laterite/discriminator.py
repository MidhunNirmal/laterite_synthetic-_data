"""
Discriminator Network for KDE-Copula GAN.
Implements Wasserstein critic for WGAN-GP training.
"""

import torch
import torch.nn as nn
from typing import List


class Discriminator(nn.Module):
    """
    Discriminator (critic) network for WGAN-GP.
    
    Input: copula latent vector z
    Output: Wasserstein distance estimate (unbounded)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 256]
    ):
        """
        Initialize discriminator.
        
        Args:
            input_dim: Dimension of input (number of features)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
        
        # Output layer (no activation for Wasserstein)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute critic score.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Critic scores of shape (batch_size, 1)
        """
        return self.network(x)
