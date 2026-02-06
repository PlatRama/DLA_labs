from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from lab1.configs.config import CNNConfig, DistillationConfig
from lab1.models.cnn import create_cnn
from lab1.utils.data_utils import create_dataloaders
from lab1.utils.trainer import Trainer, DistillationTrainer
from lab1.utils.data_utils import create_distillation_dataloaders

from utils_for_all.checkpoint import load_checkpoint
from utils_for_all.logger import setup_logger
from utils_for_all.misc import set_seed, get_device, print_model_info


def extract_teacher_outputs(
    teacher_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Extract teacher model outputs for all samples in dataloader.

    Returns:
        Tensor of teacher outputs (logits)
    """
    teacher_model.eval()
    all_outputs = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Extracting teacher outputs'):
            images = images.to(device)
            outputs = teacher_model(images)
            all_outputs.append(outputs.cpu())
    
    return torch.cat(all_outputs, dim=0)


def save_teacher_outputs(
    teacher_checkpoint: str,
    dataset_name: str = 'cifar10',
    output_path: str = './teacher_outputs.pt',
    device: str = 'cuda'
):
    """
    Save teacher outputs for training and validation sets.
    
    Args:
        teacher_checkpoint: Path to teacher checkpoint
        dataset_name: Dataset name
        output_path: Path to save outputs
        device: Device to use
    """
    print("Loading teacher model...")
    
    # Create teacher model (large network)
    teacher_config = CNNConfig(
        in_channels=3,
        base_channels=64,
        num_blocks=4,
        channels_list=[64, 128, 256, 512],
        num_classes=10,
        use_residual=True,
        experiment_name='teacher'
    )
    
    teacher_model = create_cnn(teacher_config).to(device)
    
    # Load teacher weights
    load_checkpoint(
        filepath=teacher_checkpoint,
        model=teacher_model,
        device=device
    )
    
    print_model_info(teacher_model)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=dataset_name,
        batch_size=256,  # Larger batch size for faster extraction
        num_workers=4,
        val_split=0.1,
        augment=False  # No augmentation for teacher outputs
    )
    
    # Extract outputs
    print("Extracting teacher outputs...")
    train_outputs = extract_teacher_outputs(teacher_model, train_loader, device)
    val_outputs = extract_teacher_outputs(teacher_model, val_loader, device)
    
    # Save
    print(f"Saving to {output_path}...")
    torch.save({
        'train_outputs': train_outputs,
        'val_outputs': val_outputs,
        'teacher_config': teacher_config.__dict__
    }, output_path)
    
    print("Done!")


def train_student_baseline(
    experiment_name: str = 'student_baseline',
    base_channels: int = 32
):
    """
    Train student model with only hard labels (baseline).
    
    Args:
        experiment_name: Experiment name
        base_channels: Base number of channels (smaller than teacher)
    
    Returns:
        Test metrics
    """
    print("\n" + "=" * 80)
    print(f"Training Student Baseline: {experiment_name}")
    print("=" * 80 + "\n")
    
    # Create configuration (smaller than teacher)
    config = CNNConfig(
        # Model config (smaller network)
        in_channels=3,
        base_channels=base_channels,
        num_blocks=3,
        channels_list=[32, 64, 128],
        num_classes=10,
        dropout=0.2,
        use_residual=True,
        conv_per_block=2,
        
        # Training config
        batch_size=128,
        num_epochs=100,
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
    )
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('student_baseline', log_file=f'{config.log_dir}/train.log')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=0.1,
        augment=True
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
    
    logger.info(f"Student Baseline Test Accuracy: {test_metrics['test_acc']:.2f}%")
    
    return test_metrics


def train_student_with_distillation(
    teacher_outputs_path: str,
    temperature: float = 3.0,
    alpha: float = 0.5,
    experiment_name: str = 'student_distilled',
    base_channels: int = 32
):
    """
    Train student model with knowledge distillation.
    
    Args:
        teacher_outputs_path: Path to saved teacher outputs
        temperature: Distillation temperature
        alpha: Weight for soft loss (1-alpha for hard loss)
        experiment_name: Experiment name
        base_channels: Base number of channels
    
    Returns:
        Test metrics
    """
    print("\n" + "=" * 80)
    print(f"Training Student with Distillation: {experiment_name}")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    print("=" * 80 + "\n")
    
    # Create configuration
    config = DistillationConfig(
        # Model config (same as baseline student)
        in_channels=3,
        base_channels=base_channels,
        num_blocks=3,
        channels_list=[32, 64, 128],
        num_classes=10,
        dropout=0.2,
        use_residual=True,
        conv_per_block=2,
        
        # Distillation config
        teacher_checkpoint='',  # Not needed, we have pre-computed outputs
        temperature=temperature,
        alpha=alpha,
        
        # Training config
        batch_size=128,
        num_epochs=100,
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
    )
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger = setup_logger('student_distill', log_file=f'{config.log_dir}/train.log')
    
    # Create dataloaders with teacher outputs
    train_loader, val_loader, test_loader = create_distillation_dataloaders(
        dataset_name=config.dataset,
        teacher_outputs_path=teacher_outputs_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=0.1
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
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1
    )
    
    # Create distillation trainer
    trainer = DistillationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        temperature=temperature,
        alpha=alpha
    )
    
    # Train
    test_metrics = trainer.train()
    
    logger.info(f"Student Distillation Test Accuracy: {test_metrics['test_acc']:.2f}%")
    
    return test_metrics


def run_distillation_experiments():
    """
    Run complete distillation experiment:
    1. Train baseline student
    2. Extract teacher outputs
    3. Train student with distillation
    4. Compare results
    """
    print("\n" + "=" * 80)
    print("Exercise 2.2: Knowledge Distillation Experiments")
    print("=" * 80 + "\n")
    
    # Step 1: Train baseline student
    print("Step 1: Training baseline student (hard labels only)...")
    baseline_metrics = train_student_baseline(
        experiment_name='student_baseline',
        base_channels=32
    )
    
    # Step 2: Extract teacher outputs (assumes teacher is already trained)
    teacher_checkpoint = './checkpoints/teacher_model/best.pt'
    teacher_outputs_path = './teacher_outputs.pt'
    
    if not Path(teacher_outputs_path).exists():
        print("\nStep 2: Extracting teacher outputs...")
        if Path(teacher_checkpoint).exists():
            save_teacher_outputs(
                teacher_checkpoint=teacher_checkpoint,
                dataset_name='cifar10',
                output_path=teacher_outputs_path
            )
        else:
            print(f"ERROR: Teacher checkpoint not found at {teacher_checkpoint}")
            print("Please train teacher model first using experiment_cnn.py --experiment teacher")
            return
    else:
        print("\nStep 2: Using existing teacher outputs...")
    
    # Step 3: Train student with distillation
    print("\nStep 3: Training student with distillation...")
    distill_metrics = train_student_with_distillation(
        teacher_outputs_path=teacher_outputs_path,
        temperature=3.0,
        alpha=0.5,
        experiment_name='student_distilled',
        base_channels=32
    )
    
    # Step 4: Compare results
    print("\n" + "=" * 80)
    print("Distillation Results Comparison")
    print("-" * 80)
    print(f"{'Method':<30} {'Test Accuracy':<15} {'Improvement':<15}")
    print("-" * 80)
    baseline_acc = baseline_metrics['test_acc']
    distill_acc = distill_metrics['test_acc']
    improvement = distill_acc - baseline_acc
    print(f"{'Student (baseline)':<30} {baseline_acc:<15.2f} {'-':<15}")
    print(f"{'Student (distilled)':<30} {distill_acc:<15.2f} {improvement:+.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Knowledge Distillation Experiments')
    parser.add_argument('--extract-teacher', action='store_true',
                       help='Extract teacher outputs')
    parser.add_argument('--teacher-checkpoint', type=str,
                       default='./checkpoints/teacher_model/best.pt',
                       help='Path to teacher checkpoint')
    parser.add_argument('--run-all', action='store_true',
                       help='Run complete distillation experiment')
    
    args = parser.parse_args()
    
    if args.extract_teacher:
        save_teacher_outputs(
            teacher_checkpoint=args.teacher_checkpoint,
            dataset_name='cifar10',
            output_path='./teacher_outputs.pt'
        )
    elif args.run_all:
        run_distillation_experiments()
    else:
        print("Use --run-all to run complete distillation experiment")
        print("Or use --extract-teacher to extract teacher outputs")
