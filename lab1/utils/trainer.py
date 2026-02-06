"""
Training loop and utilities for Lab1.
Uses the provided utils (checkpoint, logger, metrics, misc).
"""
import logging
import os
# Import provided utilities
import sys
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_for_all.checkpoint import CheckpointManager
from utils_for_all.logger import TensorBoardLogger, WandBLogger
from utils_for_all.metrics import MetricsTracker, compute_accuracy
from utils_for_all.misc import count_parameters, format_params, get_lr, Timer

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for classification tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Any = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            config: Configuration object
            device: Device to use
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            max_keep=2,
            save_every=config.save_every,
            metric_name='val_loss',
            mode='min'
        )
        
        # Loggers
        self.tb_logger = TensorBoardLogger(
            log_dir=config.log_dir,
            enabled=config.use_tensorboard
        )
        
        self.wandb_logger = None
        if config.use_wandb:
            wandb_config = {
                'model': model.__class__.__name__,
                'dataset': config.dataset,
                'batch_size': config.batch_size,
                'lr': config.lr,
                'num_epochs': config.num_epochs
            }
            self.wandb_logger = WandBLogger(
                project=config.wandb_project,
                name=config.experiment_name,
                config=wandb_config,
                enabled=True
            )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        logger.info(f"Trainer initialized for {config.experiment_name}")
        logger.info(f"Model parameters: {format_params(count_parameters(model))}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            acc_top1, acc_top5 = compute_accuracy(outputs, labels, topk=(1, 5))
            
            # Update metrics
            metric_tracker.update({
                'train_loss': loss.item(),
                'train_acc': acc_top1,
                'train_acc_top5': acc_top5
            }, n=images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc_top1:.2f}%"
            })
            
            # Log to tensorboard
            if self.global_step % self.config.log_every == 0:
                self.tb_logger.add_scalar('train/loss', loss.item(), self.global_step)
                self.tb_logger.add_scalar('train/acc', acc_top1, self.global_step)
                self.tb_logger.add_scalar('train/lr', get_lr(self.optimizer), self.global_step)
            
            self.global_step += 1
        
        return metric_tracker.get_averages()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metric_tracker = MetricsTracker()
        
        for images, labels in tqdm(self.val_loader, desc='Validation', leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Compute accuracy
            acc_top1, acc_top5 = compute_accuracy(outputs, labels, topk=(1, 5))
            
            # Update metrics
            metric_tracker.update({
                'val_loss': loss.item(),
                'val_acc': acc_top1,
                'val_acc_top5': acc_top5
            }, n=images.size(0))
        
        return metric_tracker.get_averages()
    
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Test the model."""
        self.model.eval()
        metric_tracker = MetricsTracker()
        
        for images, labels in tqdm(self.test_loader, desc='Testing'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Compute accuracy
            acc_top1, acc_top5 = compute_accuracy(outputs, labels, topk=(1, 5))
            
            # Update metrics
            metric_tracker.update({
                'test_loss': loss.item(),
                'test_acc': acc_top1,
                'test_acc_top5': acc_top5
            }, n=images.size(0))
        
        return metric_tracker.get_averages()
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info(f"Starting training: {self.config.experiment_name}")
        logger.info("=" * 80)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            with Timer(f"Epoch {epoch + 1}/{self.config.num_epochs}"):
                # Train
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = self.validate()
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Log metrics
                all_metrics = {**train_metrics, **val_metrics}
                logger.info(f"Epoch {epoch + 1}: " +
                          ", ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()]))
                
                # TensorBoard logging
                for key, value in val_metrics.items():
                    self.tb_logger.add_scalar(f'val/{key.replace("val_", "")}', value, epoch)
                
                # WandB logging
                if self.wandb_logger:
                    self.wandb_logger.log(all_metrics, step=epoch)
                
                # Save checkpoint
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                if (epoch + 1) % self.config.save_every == 0 or is_best:
                    self.checkpoint_manager.save(
                        step=self.global_step,
                        epoch=epoch,
                        metrics=all_metrics,
                        is_scheduled=False
                    )
        
        # Load best model and test
        logger.info("=" * 80)
        logger.info("Training complete! Loading best model for testing...")
        self.checkpoint_manager.load_best(device=self.device)
        
        test_metrics = self.test()
        logger.info("Test results: " +
                   ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))
        
        if self.wandb_logger:
            self.wandb_logger.log(test_metrics, step=self.config.num_epochs)
            self.wandb_logger.finish()
        
        self.tb_logger.close()
        
        logger.info("=" * 80)
        
        return test_metrics


class DistillationTrainer(Trainer):
    """
    Trainer for knowledge distillation.
    Extends base Trainer with distillation loss.
    """
    
    def __init__(self, *args, temperature: float = 3.0, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
        logger.info(f"Distillation: T={temperature}, alpha={alpha}")
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs (soft labels)
            labels: Hard labels
        
        Returns:
            Combined loss
        """
        # Hard loss (cross entropy with true labels)
        hard_loss = self.criterion(student_logits, labels)
        
        # Soft loss (KL divergence with teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with distillation."""
        self.model.train()
        metric_tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, labels, teacher_outputs) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            teacher_outputs = teacher_outputs.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            student_outputs = self.model(images)
            
            # Compute distillation loss
            loss = self.compute_distillation_loss(student_outputs, teacher_outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            acc_top1, acc_top5 = compute_accuracy(student_outputs, labels, topk=(1, 5))
            
            # Update metrics
            metric_tracker.update({
                'train_loss': loss.item(),
                'train_acc': acc_top1,
                'train_acc_top5': acc_top5
            }, n=images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc_top1:.2f}%"
            })
            
            # Log to tensorboard
            if self.global_step % self.config.log_every == 0:
                self.tb_logger.add_scalar('train/loss', loss.item(), self.global_step)
                self.tb_logger.add_scalar('train/acc', acc_top1, self.global_step)
            
            self.global_step += 1
        
        return metric_tracker.get_averages()
