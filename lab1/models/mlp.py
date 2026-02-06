import torch
import torch.nn as nn
from typing import List

class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        
        self.bn1 = nn.BatchNorm1d(hidden_features) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(in_features) if use_batch_norm else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        return out


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with optional residual connections.
    
    Args:
        input_size: Input feature size (e.g., 784 for MNIST)
        hidden_dims: List of hidden dimensions
        num_classes: Number of output classes
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'gelu', 'leaky_relu')
        use_residual: Whether to use residual connections
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: List[int] = [512, 512, 512],
        num_classes: int = 10,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        use_residual: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Hidden blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Adaptive layer to change dimensions
            if hidden_dims[i] != hidden_dims[i + 1]:
                adaptive = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            else:
                adaptive = nn.Identity()
            
            block = MLPBlock(
                in_features=hidden_dims[i],
                hidden_features=hidden_dims[i] * 2,  # Expansion factor
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                activation=activation
            )
            
            self.blocks.append(nn.ModuleDict({
                'block': block,
                'adaptive': adaptive
            }))
        
        # Classification head
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through blocks
        for block_dict in self.blocks:
            identity = x
            
            # Apply block
            out = block_dict['block'](x)
            
            # Add residual connection if enabled
            if self.use_residual:
                out = out + identity
            
            # Apply adaptive layer
            x = block_dict['adaptive'](out)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_features(self, x):
        """
        Extract features before classification head.
        Useful for fine-tuning and feature extraction.
        """
        x = x.view(x.size(0), -1)
        x = self.input_proj(x)
        
        for block_dict in self.blocks:
            identity = x
            out = block_dict['block'](x)
            if self.use_residual:
                out = out + identity
            x = block_dict['adaptive'](out)
        
        return x


def create_mlp(config) -> MLP:
    """
    Create MLP model from configuration.
    
    Args:
        config: MLPConfig object
    
    Returns:
        MLP model
    """
    return MLP(
        input_size=config.input_size,
        hidden_dims=config.hidden_dims,
        num_classes=config.num_classes,
        dropout=config.dropout,
        use_batch_norm=config.use_batch_norm,
        activation=config.activation,
        use_residual=config.use_residual
    )