import torch
import torch.nn as nn
from typing import Optional


class SimpleCNN(nn.Module):
    """
    Simple CNN classifier for CIFAR-10/100.
    Used as baseline and for OOD detection experiments.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 -> 4
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Linear(base_channels * 4, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract features before classifier."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Autoencoder(nn.Module):
    """
    Autoencoder for OOD detection.
    Uses encoder from pre-trained classifier + trainable decoder.
    """
    
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        latent_dim: int = 256,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        # Encoder (from pre-trained classifier)
        if encoder is not None:
            self.encoder = encoder
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            # Default encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),  # 32 -> 16
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16 -> 8
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8 -> 4
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsample from latent
            nn.ConvTranspose2d(latent_dim, 128, 4, stride=1, padding=0),  # 1 -> 4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 16 -> 32
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self._init_decoder()
    
    def _init_decoder(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Reshape if needed
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon
    
    def encode(self, x):
        """Get latent representation."""
        return self.encoder(x)


def create_autoencoder_from_classifier(
    classifier: nn.Module,
    latent_dim: int = 256,
    freeze_encoder: bool = True
) -> Autoencoder:
    """
    Create autoencoder using features from pre-trained classifier.
    
    Args:
        classifier: Pre-trained classifier
        latent_dim: Latent dimension
        freeze_encoder: Whether to freeze encoder
    
    Returns:
        Autoencoder model
    """
    # Extract encoder from classifier
    if hasattr(classifier, 'features'):
        encoder = nn.Sequential(
            classifier.features,
            classifier.avgpool
        )
    else:
        # Fallback: use entire classifier except final layer
        modules = list(classifier.children())[:-1]
        encoder = nn.Sequential(*modules)
    
    autoencoder = Autoencoder(
        encoder=encoder,
        latent_dim=latent_dim,
        freeze_encoder=freeze_encoder
    )
    
    return autoencoder