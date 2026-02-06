"""
Exercise 1.1 and 1.2: MLP experiments
Train MLPs with and without residual connections on MNIST.
"""

import torch.optim as optim

from lab1.configs.config import MLPConfig
from lab1.models.mlp import create_mlp
from lab1.utils.data_utils import create_dataloaders
from lab1.utils.trainer import Trainer
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed, get_device, print_model_info


def run_mlp_experiment(
    depth: int,
    use_residual: bool,
    hidden_dim: int = 512,
    experiment_name: str = None
):
    """
    Run MLP experiment with specified depth and residual configuration.
    
    Args:
        depth: Number of hidden layers
        use_residual: Whether to use residual connections
        hidden_dim: Hidden layer dimension
        experiment_name: Name for this experiment
    """
    # Setup experiment name
    if experiment_name is None:
        res_str = "residual" if use_residual else "no_residual"
        experiment_name = f"mlp_depth{depth}_{res_str}"
    
    # Create configuration
    config = MLPConfig(
        # Model config
        input_size=784,
        hidden_dims=[hidden_dim] * depth,
        num_classes=10,
        dropout=0.2,
        use_batch_norm=True,
        activation='relu',
        use_residual=use_residual,
        
        # Training config
        batch_size=128,
        num_epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
        
        # Dataset
        dataset='mnist',
        num_workers=4,
        
        # Experiment
        experiment_name=experiment_name,
        checkpoint_dir=f'./checkpoints/lab1/{experiment_name}',
        log_dir=f'./logs/lab1/{experiment_name}',
        
        # Logging
        use_tensorboard=True,
        use_wandb=False,
        log_every=50,
        save_every=5,
    )
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('mlp_experiment', log_file=f'{config.log_dir}/train.log')
    
    logger.info("=" * 80)
    logger.info(f"MLP Experiment: {experiment_name}")
    logger.info(f"Depth: {depth}, Residual: {use_residual}")
    logger.info("=" * 80)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=0.1,
        augment=False  # No augmentation for MNIST
    )
    
    # Create model
    model = create_mlp(config)
    print_model_info(model)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # Train
    test_metrics = trainer.train()
    
    logger.info("=" * 80)
    logger.info(f"Experiment {experiment_name} completed!")
    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.2f}%")
    logger.info("=" * 80)
    
    return test_metrics


def compare_depths():
    """
    Exercise 1.1: Compare MLPs of different depths without residual connections.
    Verify that deeper networks don't always perform better.
    """
    print("\n" + "=" * 80)
    print("Exercise 1.1: Comparing MLP depths WITHOUT residual connections")
    print("=" * 80 + "\n")
    
    depths = [2, 4, 6, 8, 10]
    results = {}
    
    for depth in depths:
        metrics = run_mlp_experiment(
            depth=depth,
            use_residual=False,
            hidden_dim=512
        )
        results[f'depth_{depth}'] = metrics['test_acc']
    
    # Print comparison
    print("\n" + "=" * 80)
    print("Results without residual connections:")
    print("-" * 80)
    for depth in depths:
        acc = results[f'depth_{depth}']
        print(f"Depth {depth:2d}: {acc:.2f}% accuracy")
    print("=" * 80 + "\n")


def compare_with_residual():
    """
    Exercise 1.2: Compare MLPs with and without residual connections.
    Verify that residual connections help with deeper networks.
    """
    print("\n" + "=" * 80)
    print("Exercise 1.2: Comparing WITH and WITHOUT residual connections")
    print("=" * 80 + "\n")
    
    depths = [6, 8, 10]
    results = {}
    
    for depth in depths:
        # Without residual
        print(f"\nTraining depth {depth} WITHOUT residual...")
        metrics_no_res = run_mlp_experiment(
            depth=depth,
            use_residual=False,
            hidden_dim=512
        )
        results[f'depth_{depth}_no_res'] = metrics_no_res['test_acc']
        
        # With residual
        print(f"\nTraining depth {depth} WITH residual...")
        metrics_res = run_mlp_experiment(
            depth=depth,
            use_residual=True,
            hidden_dim=512
        )
        results[f'depth_{depth}_res'] = metrics_res['test_acc']
    
    # Print comparison
    print("\n" + "=" * 80)
    print("Comparison: Residual vs No Residual")
    print("-" * 80)
    print(f"{'Depth':<10} {'No Residual':<15} {'With Residual':<15} {'Improvement':<15}")
    print("-" * 80)
    for depth in depths:
        no_res = results[f'depth_{depth}_no_res']
        res = results[f'depth_{depth}_res']
        improvement = res - no_res
        print(f"{depth:<10} {no_res:<15.2f} {res:<15.2f} {improvement:+.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run Exercise 1.1: Compare different depths without residual
    compare_depths()
    
    # Run Exercise 1.2: Compare with and without residual connections
    compare_with_residual()
