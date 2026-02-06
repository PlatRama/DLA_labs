import torch.optim as optim

from lab1.configs.config import CNNConfig
from lab1.models.cnn import create_cnn
from lab1.utils.data_utils import create_dataloaders
from lab1.utils.trainer import Trainer
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed, get_device, print_model_info


def run_cnn_experiment(
    depth: int,
    use_residual: bool,
    base_channels: int = 64,
    experiment_name: str = None
):
    """
    Run CNN experiment with specified depth and residual configuration.
    
    Args:
        depth: Number of stages (each stage has multiple blocks)
        use_residual: Whether to use residual connections
        base_channels: Base number of channels
        experiment_name: Name for this experiment
    """
    # Setup experiment name
    if experiment_name is None:
        res_str = "residual" if use_residual else "no_residual"
        experiment_name = f"cnn_depth{depth}_{res_str}"
    
    # Create channel progression
    if depth == 3:
        channels_list = [64, 128, 256]
    elif depth == 4:
        channels_list = [64, 128, 256, 512]
    elif depth == 5:
        channels_list = [64, 128, 256, 512, 512]
    else:
        # Dynamic generation for other depths
        channels_list = [base_channels * (2 ** i) for i in range(depth)]
        # Cap at 512
        channels_list = [min(c, 512) for c in channels_list]
    
    # Create configuration
    config = CNNConfig(
        # Model config
        in_channels=3,
        base_channels=base_channels,
        num_blocks=depth,
        channels_list=channels_list,
        num_classes=10,
        dropout=0.2,
        use_residual=use_residual,
        conv_per_block=2,
        
        # Training config
        batch_size=128,
        num_epochs=20,
        lr=1e-3,
        weight_decay=5e-4,
        
        # Dataset
        dataset='cifar10',
        num_workers=4,
        
        # Experiment
        experiment_name=experiment_name,
        checkpoint_dir=f'./checkpoints/{experiment_name}',
        log_dir=f'./logs/{experiment_name}',
        
        # Logging
        use_tensorboard=True,
        use_wandb=False,
        log_every=50,
        save_every=10,
        
        # Early stopping
        early_stopping=True,
        early_stopping_patience=15
    )
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('cnn_experiment', log_file=f'{config.log_dir}/train.log')
    
    logger.info("=" * 80)
    logger.info(f"CNN Experiment: {experiment_name}")
    logger.info(f"Depth: {depth}, Residual: {use_residual}")
    logger.info(f"Channels: {channels_list}")
    logger.info("=" * 80)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=0.1,
        augment=True  # Data augmentation for CIFAR
    )
    
    # Create model
    model = create_cnn(config)
    print_model_info(model)
    
    # Create optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay
    )
    
    # Multi-step learning rate decay
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1
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


def compare_cnn_depths():
    """
    Compare CNNs of different depths WITHOUT residual connections.
    Similar to Exercise 1.1 but with CNNs.
    """
    print("\n" + "=" * 80)
    print("Exercise 1.3a: Comparing CNN depths WITHOUT residual connections")
    print("=" * 80 + "\n")
    
    depths = [1, 5, 10]
    results = {}
    
    for depth in depths:
        metrics = run_cnn_experiment(
            depth=depth,
            use_residual=False,
            base_channels=64
        )
        results[f'depth_{depth}'] = metrics['test_acc']
    
    # Print comparison
    print("\n" + "=" * 80)
    print("Results without residual connections:")
    print("-" * 80)
    for depth in depths:
        acc = results[f'depth_{depth}']
        print(f"Depth {depth}: {acc:.2f}% accuracy")
    print("=" * 80 + "\n")


def compare_cnn_with_residual():
    """
    Compare CNNs WITH and WITHOUT residual connections.
    Verify that residual connections help train deeper networks.
    """
    print("\n" + "=" * 80)
    print("Exercise 1.3b: Comparing WITH and WITHOUT residual connections")
    print("=" * 80 + "\n")
    
    depths = [1, 5, 10]
    results = {}
    
    for depth in depths:
        # Without residual
        print(f"\nTraining depth {depth} WITHOUT residual...")
        metrics_no_res = run_cnn_experiment(
            depth=depth,
            use_residual=False,
            base_channels=64
        )
        results[f'depth_{depth}_no_res'] = metrics_no_res['test_acc']
        
        # With residual
        print(f"\nTraining depth {depth} WITH residual...")
        metrics_res = run_cnn_experiment(
            depth=depth,
            use_residual=True,
            base_channels=64
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


def train_teacher_model():
    """
    Train a large teacher model for distillation experiments.
    This will be used in Exercise 2.2.
    """
    print("\n" + "=" * 80)
    print("Training Teacher Model for Distillation")
    print("=" * 80 + "\n")
    
    metrics = run_cnn_experiment(
        depth=4,
        use_residual=True,
        base_channels=64,
        experiment_name='teacher_model'
    )
    
    print(f"\nTeacher model trained with {metrics['test_acc']:.2f}% accuracy")
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CNN Experiments for Lab1')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'depths', 'residual', 'teacher'],
                       help='Which experiment to run')
    
    args = parser.parse_args()
    
    """if args.experiment == 'all' or args.experiment == 'depths':
        # Compare different depths without residual
        compare_cnn_depths()"""
    
    if args.experiment == 'all' or args.experiment == 'residual':
        # Compare with and without residual connections
        compare_cnn_with_residual()
    
    #if args.experiment == 'teacher':
        # Train teacher model
        #train_teacher_model()
