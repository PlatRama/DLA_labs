import torch
import torch.optim as optim

from lab4.configs.config import AutoencoderConfig
from lab4.models.ood_models import SimpleCNN, create_autoencoder_from_classifier
from lab4.utils.data_utils import create_id_dataloaders
from lab4.utils.trainer import AutoencoderTrainer
from utils_for_all.checkpoint import load_checkpoint
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed, get_device


def train_autoencoder(
    classifier_checkpoint: str = './checkpoints/lab4/classifier_baseline/best.pt',
    freeze_encoder: bool = True,
    num_epochs: int = 50,
    experiment_name: str = 'autoencoder'
):
    """
    Train autoencoder for OOD detection.
    
    Args:
        classifier_checkpoint: Pre-trained classifier checkpoint
        freeze_encoder: Whether to freeze encoder
        num_epochs: Number of epochs
        experiment_name: Experiment name
    """
    config = AutoencoderConfig(
        encoder_checkpoint=classifier_checkpoint,
        freeze_encoder=freeze_encoder,
        latent_dim=256,
        num_epochs=num_epochs,
        lr=1e-3,
        batch_size=128,
        reconstruction_loss='mse',
        use_augmentation=False,
        experiment_name=experiment_name,
        checkpoint_dir=f'./checkpoints/lab4/{experiment_name}',
        log_dir=f'./logs/lab4/{experiment_name}',
        save_every=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('autoencoder', log_file=f'{config.log_dir}/train.log')
    
    logger.info("=" * 80)
    logger.info(f"Training Autoencoder")
    logger.info(f"Freeze Encoder: {freeze_encoder}")
    logger.info("=" * 80)
    
    # Load pre-trained classifier
    logger.info(f"Loading classifier from {classifier_checkpoint}")
    classifier = SimpleCNN(num_classes=10)
    load_checkpoint(
        classifier_checkpoint,
        model=classifier,
        device=device,
        strict=True
    )
    
    # Create autoencoder from classifier
    autoencoder = create_autoencoder_from_classifier(
        classifier=classifier,
        latent_dim=config.latent_dim,
        freeze_encoder=freeze_encoder
    )
    
    # Dataloaders (only ID data)
    train_loader, val_loader, _ = create_id_dataloaders(
        dataset='cifar10',
        batch_size=config.batch_size,
        num_workers=4,
        augment=config.use_augmentation,
        root='./data'
    )
    
    # Optimizer (only decoder parameters if encoder frozen)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, autoencoder.parameters()),
        lr=config.lr
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )
    
    # Trainer
    trainer = AutoencoderTrainer(
        model=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # Train
    results = trainer.train()
    
    logger.info(f"\nAutoencoder training completed!")
    logger.info(f"Best Validation Loss: {results['best_val_loss']:.6f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--checkpoint', type=str,
                       default='./checkpoints/lab4/classifier_baseline/best.pt',
                       help='Classifier checkpoint')
    parser.add_argument('--freeze-encoder', action='store_true', default=True,
                       help='Freeze encoder weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--experiment-name', type=str, default='autoencoder')
    
    args = parser.parse_args()
    
    results = train_autoencoder(
        classifier_checkpoint=args.checkpoint,
        freeze_encoder=args.freeze_encoder,
        num_epochs=args.epochs,
        experiment_name=args.experiment_name
    )
    
    print(f"\nBest Validation Loss: {results['best_val_loss']:.6f}")
