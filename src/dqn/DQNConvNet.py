"""
Implement the ConvNet for DQN.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

class DQNConvNet(nn.Module):
    """
    The ConvNet structure is taken from Mnih et al.
    """
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1)) # (B, C, H, W) -> (B, H, W, C)
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=3)(x)
        return x
