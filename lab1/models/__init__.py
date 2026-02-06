"""
Lab1 Rewrite - Deep Learning Applications
Models module
"""
from .mlp import MLP, create_mlp
from .cnn import CNN, create_cnn, ResidualBlock

__all__ = ['MLP', 'create_mlp', 'CNN', 'create_cnn', 'ResidualBlock']
