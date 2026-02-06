"""
Convolutional Neural Network models for Lab1.
"""
import torch
import torch.nn as nn
from typing import List

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection (identity or projection)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.use_residual:
            identity = self.skip(identity)
            out = out + identity
        
        out = self.relu(out)
        
        return out


class ConvBlock(nn.Module):
    """
    Basic convolutional block without residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network with optional residual connections.
    
    Args:
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
        base_channels: Base number of channels
        num_blocks: Number of blocks
        channels_list: List of channels for each block
        num_classes: Number of output classes
        dropout: Dropout rate
        use_residual: Whether to use residual connections
        conv_per_block: Number of residual blocks per stage
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 3,
        channels_list: List[int] = [64, 128, 256],
        num_classes: int = 10,
        dropout: float = 0.2,
        use_residual: bool = True,
        conv_per_block: int = 2
    ):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, base_channels,
                kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Build blocks
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i, out_channels in enumerate(channels_list):
            # Determine stride for first block (downsample if needed)
            stride = 2 if i > 0 else 1
            
            # Create blocks
            block_list = []
            for j in range(conv_per_block):
                if j == 0:
                    # First block in stage (may downsample)
                    if use_residual:
                        block = ResidualBlock(
                            current_channels, out_channels,
                            stride=stride, use_residual=True, dropout=dropout
                        )
                    else:
                        block = ConvBlock(
                            current_channels, out_channels,
                            stride=stride, dropout=dropout
                        )
                else:
                    # Remaining blocks (no downsampling)
                    if use_residual:
                        block = ResidualBlock(
                            out_channels, out_channels,
                            stride=1, use_residual=True, dropout=dropout
                        )
                    else:
                        block = ConvBlock(
                            out_channels, out_channels,
                            stride=1, dropout=dropout
                        )
                
                block_list.append(block)
                current_channels = out_channels
            
            self.blocks.append(nn.Sequential(*block_list))
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels_list[-1], num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.avgpool(x)
        logits = self.classifier(x)
        
        return logits
    
    def get_features(self, x):
        """
        Extract features before classification head.
        Useful for fine-tuning and feature extraction.
        """
        x = self.conv1(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features


def create_cnn(config) -> CNN:
    """
    Create CNN model from configuration.
    
    Args:
        config: CNNConfig object
    
    Returns:
        CNN model
    """
    return CNN(
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        channels_list=config.channels_list,
        num_classes=config.num_classes,
        dropout=config.dropout,
        use_residual=config.use_residual,
        conv_per_block=config.conv_per_block
    )
