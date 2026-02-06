import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SentimentClassifier(nn.Module):
    """
    Simple sentiment classifier using a pretrained transformer backbone.
    
    Uses [CLS] token output from the backbone + classification head.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert/distilbert-base-uncased",
        hidden_dim: int = 768,
        classifier_hidden: int = 256,
        num_labels: int = 2,
        dropout: float = 0.2,
        train_backbone: bool = False
    ):
        """
        Args:
            model_name: Name of pretrained model
            hidden_dim: Hidden dimension of backbone
            classifier_hidden: Hidden dimension of classifier
            num_labels: Number of output classes
            dropout: Dropout rate
            train_backbone: Whether to train the backbone
        """
        super().__init__()
        
        self.model_name = model_name
        self.train_backbone = train_backbone
        
        # Load backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone if needed
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(classifier_hidden),
            nn.Linear(classifier_hidden, num_labels)
        )
        
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            Logits [batch, num_labels]
        """
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for training."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.train_backbone = True


class BinarySentimentClassifier(nn.Module):
    """
    Binary sentiment classifier (positive/negative).
    Uses sigmoid output for binary classification.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert/distilbert-base-uncased",
        hidden_dim: int = 768,
        classifier_hidden: int = 256,
        dropout: float = 0.2,
        train_backbone: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.train_backbone = train_backbone
        
        # Load backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone if needed
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Binary classification head (outputs single logit)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(classifier_hidden),
            nn.Linear(classifier_hidden, 1)  # Single output for binary
        )
        
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass.
        
        Returns:
            Logits [batch, 1] - use with BCEWithLogitsLoss
        """
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Binary classification
        logits = self.classifier(cls_output)
        
        return logits.squeeze(-1)  # [batch]


def create_model_from_config(config, binary: bool = False):
    """
    Create model from configuration.
    
    Args:
        config: Configuration object
        binary: If True, use binary classifier; else multi-class
    
    Returns:
        Model
    """
    if binary:
        return BinarySentimentClassifier(
            model_name=config.model_name,
            hidden_dim=config.hidden_dim,
            classifier_hidden=config.classifier_hidden,
            dropout=config.dropout,
            train_backbone=config.train_backbone
        )
    else:
        return SentimentClassifier(
            model_name=config.model_name,
            hidden_dim=config.hidden_dim,
            classifier_hidden=config.classifier_hidden,
            num_labels=2,
            dropout=config.dropout,
            train_backbone=config.train_backbone
        )