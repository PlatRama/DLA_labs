from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class BaseConfig:
    """Base configuration for Lab3 experiments."""
    # Device and reproducibility
    device: str = 'cuda'
    seed: int = 42
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0
    
    # Model
    model_name: str = "distilbert/distilbert-base-uncased"
    
    # Logging and checkpointing
    experiment_name: str = 'baseline'
    checkpoint_dir: str = './checkpoints/lab3'
    log_dir: str = './logs/lab3'
    save_every: int = 1  # Save every N epochs
    log_every: int = 50

    
    # Logging
    use_tensorboard: bool = True


@dataclass
class BaselineConfig(BaseConfig):
    """Configuration for baseline experiments (feature extraction + classifier)."""
    # Model architecture
    hidden_dim: int = 768  # DistilBERT hidden size
    classifier_hidden: int = 256
    dropout: float = 0.2
    train_backbone: bool = False  # Freeze backbone by default
    
    # Experiment
    experiment_name: str = 'baseline_frozen'


@dataclass
class FineTuningConfig(BaseConfig):
    """Configuration for full fine-tuning experiments."""
    # Model config
    num_labels: int = 2
    hidden_dropout_prob: float = 0.2
    classifier_dropout: float = 0.3
    
    # Training (fine-tuning typically needs lower LR)
    lr: float = 2e-5
    warmup_steps: int = 500
    
    # Experiment
    experiment_name: str = 'finetuning_full'


@dataclass  
class LoRAConfig(BaseConfig):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""
    # LoRA parameters
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_lin", "k_lin", "v_lin"]
    )
    
    # Model config
    num_labels: int = 2
    hidden_dropout_prob: float = 0.2
    classifier_dropout: float = 0.3
    
    # Training
    lr: float = 1e-4
    
    # Experiment
    experiment_name: str = 'finetuning_lora'


@dataclass
class CLIPConfig(BaseConfig):
    """Configuration for CLIP experiments (Exercise 3.2)."""
    # CLIP model
    model_name: str = "openai/clip-vit-base-patch16"
    
    # Dataset
    dataset_name: str = "imagenette"
    image_size: int = 224
    
    # Zero-shot settings
    class_names: List[str] = field(default_factory=list)
    prompt_template: str = "a photo of a {}"
    
    # Fine-tuning settings
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = True
    
    # Training
    lr: float = 1e-5
    batch_size: int = 32
    
    experiment_name: str = 'clip_finetuning'


def get_config(experiment_type: str, **kwargs) -> BaseConfig:
    """
    Factory function to get appropriate configuration.
    
    Args:
        experiment_type: Type of experiment
        **kwargs: Additional parameters to override defaults
    
    Returns:
        Configuration object
    """
    config_map = {
        'baseline': BaselineConfig,
        'finetuning': FineTuningConfig,
        'lora': LoRAConfig,
        'clip': CLIPConfig,
    }
    
    if experiment_type not in config_map:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    return config_map[experiment_type](**kwargs)
