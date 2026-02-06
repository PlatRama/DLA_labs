import torch
import torch.optim as optim
from lab4.configs.config import RobustTrainingConfig
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed, get_device
from lab4.models.ood_models import SimpleCNN
from lab4.utils.data_utils import create_id_dataloaders
from lab4.utils.trainer import ClassifierTrainer


def train_robust_classifier(
    num_epochs: int = 100,
    adv_ratio: float = 0.5,
    train_epsilon: float = 0.031,
    experiment_name: str = 'robust_classifier'
):
    """
    Train robust classifier with adversarial training.
    
    Args:
        num_epochs: Number of epochs
        adv_ratio: Ratio of adversarial examples
        train_epsilon: Epsilon for training attacks
        experiment_name: Experiment name
    """
    config = RobustTrainingConfig(
        # Training
        num_epochs=num_epochs,
        lr=0.1,
        batch_size=128,
        
        # Adversarial training
        adv_train=True,
        adv_ratio=adv_ratio,
        train_epsilon=train_epsilon,
        train_pgd_steps=7,
        train_pgd_alpha=0.007,
        
        # Experiment
        experiment_name=experiment_name,
        checkpoint_dir=f'./checkpoints/lab4/{experiment_name}',
        log_dir=f'./logs/lab4/{experiment_name}',
        save_every=10,
        
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )
    
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('robust', log_file=f'{config.log_dir}/train.log')
    
    logger.info("=" * 80)
    logger.info(f"Robust Training: {experiment_name}")
    logger.info(f"Adversarial Ratio: {adv_ratio}")
    logger.info("=" * 80)
    
    # Dataloaders
    train_loader, val_loader, test_loader = create_id_dataloaders(
        dataset='cifar10',
        batch_size=config.batch_size,
        num_workers=4,
        augment=True,
        root='./data'
    )
    
    # Model
    model = SimpleCNN(num_classes=10, base_channels=64)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_schedule,
        gamma=0.1
    )
    
    # Trainer
    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        adversarial_training=True
    )
    
    # Train
    test_metrics = trainer.train()
    
    logger.info(f"\nRobust training completed!")
    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.2f}%")
    
    return test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust Training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--adv-ratio', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.031)
    parser.add_argument('--experiment-name', type=str, default='robust_classifier')
    
    args = parser.parse_args()
    
    metrics = train_robust_classifier(
        num_epochs=args.epochs,
        adv_ratio=args.adv_ratio,
        train_epsilon=args.epsilon,
        experiment_name=args.experiment_name
    )
    
    print(f"\nFinal Test Accuracy: {metrics['test_acc']:.2f}%")
