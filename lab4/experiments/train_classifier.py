import sys
from pathlib import Path

import torch
import torch.optim as optim

from configs.config import BaseConfig
from models.ood_models import SimpleCNN
from utils.data_utils import create_id_dataloaders
from utils.trainer import ClassifierTrainer
from logger import setup_logger
from misc import set_seed, get_device


def train_classifier(
    dataset: str = 'cifar10',
    num_classes: int = 10,
    num_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    experiment_name: str = 'classifier_baseline'
):
    """
    Train baseline classifier.
    
    Args:
        dataset: Dataset name
        num_classes: Number of classes
        num_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        experiment_name: Experiment name
    """
    # Create configuration
    config = BaseConfig(
        # Dataset
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        
        # Training
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        
        # Experiment
        experiment_name=experiment_name,
        checkpoint_dir=f'./checkpoints/lab4/{experiment_name}',
        log_dir=f'./logs/lab4/{experiment_name}',
        save_every=10,
        
        # Logging
        use_tensorboard=True,
        use_wandb=False,
        log_every=100,
        
        # Early stopping
        early_stopping=True,
        early_stopping_patience=20,
        
        # Device
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('classifier', log_file=f'{config.log_dir}/train.log')
    
    logger.info("=" * 80)
    logger.info(f"Training Classifier: {experiment_name}")
    logger.info(f"Dataset: {dataset}")
    logger.info("=" * 80)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_id_dataloaders(
        dataset=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=0.0,  # Use test as validation
        augment=True,
        root='./data'
    )
    
    # Create model
    model = SimpleCNN(
        in_channels=3,
        num_classes=num_classes,
        base_channels=64
    )
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1
    )
    
    # Create trainer
    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        adversarial_training=False
    )
    
    # Train
    test_metrics = trainer.train()
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.2f}%")
    
    return test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Baseline Classifier')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset name')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--experiment-name', type=str, default='classifier_baseline',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    metrics = train_classifier(
        dataset=args.dataset,
        num_classes=args.num_classes,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        experiment_name=args.experiment_name
    )
    
    print(f"\nFinal Test Accuracy: {metrics['test_acc']:.2f}%")
