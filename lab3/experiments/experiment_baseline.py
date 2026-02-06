import torch
import torch.optim as optim

from lab3.configs.config import BaselineConfig
from lab3.models.sentiment_models import BinarySentimentClassifier
from lab3.utils.data_utils import create_sentiment_dataloaders
from lab3.utils.trainer import SentimentTrainer
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed, get_device


def run_baseline_experiment(
    train_backbone: bool = False,
    experiment_name: str = None
):
    """
    Run baseline experiment with sentiment classifier.

    Args:
        train_backbone: Whether to train the backbone
        experiment_name: Name for experiment
    """
    # Setup experiment name
    if experiment_name is None:
        backbone_str = "trainable" if train_backbone else "frozen"
        experiment_name = f"baseline_{backbone_str}"
    
    # Create configuration
    config = BaselineConfig(
        # Model
        model_name="distilbert/distilbert-base-uncased",
        hidden_dim=768,
        classifier_hidden=256,
        dropout=0.2,
        train_backbone=train_backbone,
        
        # Training
        batch_size=64,
        num_epochs=10,
        lr=2e-5 if train_backbone else 1e-4,  # Lower LR if training backbone
        weight_decay=0.01,
        
        # Experiment
        experiment_name=experiment_name,
        checkpoint_dir=f'./checkpoints/lab3/{experiment_name}',
        log_dir=f'./logs/lab3/{experiment_name}',
        
        # Logging
        use_tensorboard=True,
        log_every=50,
        
        # Device
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('baseline', log_file=f'{config.log_dir}/train.log')
    
    logger.info("=" * 80)
    logger.info(f"Baseline Experiment: {experiment_name}")
    logger.info(f"Train Backbone: {train_backbone}")
    logger.info("=" * 80)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_sentiment_dataloaders(
        model_name=config.model_name,
        batch_size=config.batch_size,
        max_length=512,
        num_workers=4,
        use_hf_trainer=False
    )
    
    # Create model
    model = BinarySentimentClassifier(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        classifier_hidden=config.classifier_hidden,
        dropout=config.dropout,
        train_backbone=config.train_backbone
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )
    
    # Create trainer
    trainer = SentimentTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        binary=True  # Binary classification
    )
    
    # Train
    test_metrics = trainer.train()
    
    logger.info(f"\nExperiment {experiment_name} completed!")
    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.2f}%")
    logger.info(f"Test F1: {test_metrics['test_f1']:.4f}")
    
    return test_metrics


def compare_frozen_vs_trainable():
    """
    Compare baseline with frozen vs trainable backbone.
    """
    print("\n" + "=" * 80)
    print("Comparing Frozen vs Trainable Backbone")
    print("=" * 80 + "\n")
    
    # Frozen backbone
    print("1. Training with FROZEN backbone...")
    frozen_metrics = run_baseline_experiment(
        train_backbone=False,
        experiment_name='baseline_frozen'
    )
    
    # Trainable backbone
    print("\n2. Training with TRAINABLE backbone...")
    trainable_metrics = run_baseline_experiment(
        train_backbone=True,
        experiment_name='baseline_trainable'
    )
    
    # Compare
    print("\n" + "=" * 80)
    print("Results Comparison")
    print("-" * 80)
    print(f"{'Method':<30} {'Test Acc':<15} {'Test F1':<15}")
    print("-" * 80)
    print(f"{'Frozen Backbone':<30} {frozen_metrics['test_acc']:<15.2f} {frozen_metrics['test_f1']:<15.4f}")
    print(f"{'Trainable Backbone':<30} {trainable_metrics['test_acc']:<15.2f} {trainable_metrics['test_f1']:<15.4f}")
    improvement_acc = trainable_metrics['test_acc'] - frozen_metrics['test_acc']
    improvement_f1 = trainable_metrics['test_f1'] - frozen_metrics['test_f1']
    print(f"{'Improvement':<30} {improvement_acc:+<15.2f} {improvement_f1:+<15.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline Experiments')
    parser.add_argument('--train-backbone', action='store_true',
                       help='Train the backbone (else freeze it)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare frozen vs trainable')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_frozen_vs_trainable()
    else:
        metrics = run_baseline_experiment(
            train_backbone=args.train_backbone,
            experiment_name=args.experiment_name
        )
        print(f"\nFinal Test Accuracy: {metrics['test_acc']:.2f}%")
