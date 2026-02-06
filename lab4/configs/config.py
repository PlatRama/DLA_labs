from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class BaseConfig:
    """Base configuration for Lab4 experiments."""
    # Device and reproducibility
    device: str = 'cuda'
    seed: int = 42
    
    # Training parameters
    batch_size: int = 128
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.0
    
    # Dataset
    dataset: str = 'cifar10'  # ID dataset
    num_workers: int = 4
    
    # Logging and checkpointing
    experiment_name: str = 'baseline'
    checkpoint_dir: str = './checkpoints/lab4'
    log_dir: str = './logs/lab4'
    save_every: int = 5
    log_every: int = 50
    
    # Logging
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = 'dla-lab4'
    wandb_entity: Optional[str] = None


@dataclass
class OODDetectionConfig(BaseConfig):
    """Configuration for OOD detection experiments."""
    # Model checkpoint (pre-trained classifier)
    classifier_checkpoint: str = './checkpoints/lab4/classifier_best.pt'
    
    # OOD dataset
    ood_dataset: str = 'cifar100_subset'  # 'cifar100_subset', 'fakedata', 'svhn'
    ood_num_samples: int = 10000
    
    # OOD scoring method
    score_method: str = 'max_softmax'  # 'max_softmax', 'max_logit', 'energy', 'mse'
    temperature: float = 1.0  # Temperature for softmax
    
    # Metrics
    fpr_threshold: float = 0.95  # FPR@95 metric
    
    experiment_name: str = 'ood_detection'


@dataclass
class AdversarialConfig(BaseConfig):
    """Configuration for adversarial attacks."""
    # Model checkpoint
    classifier_checkpoint: str = './checkpoints/lab4/classifier_best.pt'
    
    # Attack method
    attack_method: str = 'fgsm'  # 'fgsm', 'pgd', 'bim'
    
    # FGSM parameters
    epsilon: float = 0.031  # 8/255 in normalized space
    
    # PGD parameters
    pgd_steps: int = 40
    pgd_alpha: float = 0.01  # Step size
    pgd_random_start: bool = True
    
    # Attack type
    targeted: bool = False
    target_class: Optional[int] = None
    
    # Evaluation
    num_test_samples: int = 1000
    
    experiment_name: str = 'adversarial_attack'


@dataclass
class RobustTrainingConfig(BaseConfig):
    """Configuration for adversarial training (robust classifier)."""
    # Adversarial training
    adv_train: bool = True
    adv_ratio: float = 0.5  # Ratio of adversarial examples in batch
    
    # Attack during training
    train_attack: str = 'pgd'
    train_epsilon: float = 0.031
    train_pgd_steps: int = 7  # Fewer steps during training
    train_pgd_alpha: float = 0.007
    
    # Model architecture (same as classifier)
    in_channels: int = 3
    num_classes: int = 10
    base_channels: int = 64
    
    # Training
    num_epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_schedule: List[int] = field(default_factory=lambda: [50, 75])
    
    experiment_name: str = 'robust_training'


@dataclass
class AutoencoderConfig(BaseConfig):
    """Configuration for autoencoder training (for OOD detection)."""
    # Encoder (use pre-trained classifier backbone)
    encoder_checkpoint: str = './checkpoints/lab4/classifier_best.pt'
    freeze_encoder: bool = True
    
    # Decoder architecture
    latent_dim: int = 512  # Encoder output dimension
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    
    # Training
    num_epochs: int = 50
    lr: float = 1e-3
    reconstruction_loss: str = 'mse'  # 'mse', 'l1'
    
    # Data
    use_augmentation: bool = True
    
    experiment_name: str = 'autoencoder'


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
        'ood_detection': OODDetectionConfig,
        'adversarial': AdversarialConfig,
        'robust_training': RobustTrainingConfig,
        'autoencoder': AutoencoderConfig,
    }
    
    if experiment_type not in config_map:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    return config_map[experiment_type](**kwargs)
