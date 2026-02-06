import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lab4.utils.adversarial_utils import adversarial_training_loss
from utils_for_all.checkpoint import CheckpointManager
from utils_for_all.logger import TensorBoardLogger, WandBLogger
from utils_for_all.metrics import MetricsTracker
from utils_for_all.misc import count_parameters, format_params, get_lr

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """
    Trainer for classifier (with optional adversarial training).
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
        device: str = 'cuda',
        adversarial_training: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.adversarial_training = adversarial_training
        
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
                'batch_size': config.batch_size,
                'lr': config.lr,
                'num_epochs': config.num_epochs,
                'adversarial_training': adversarial_training
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
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        
        logger.info(f"Trainer initialized for {config.experiment_name}")
        logger.info(f"Model parameters: {format_params(count_parameters(model))}")
        if adversarial_training:
            logger.info("Adversarial training enabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.adversarial_training and hasattr(self.config, 'adv_ratio'):
                # Mix clean and adversarial examples
                batch_size = images.size(0)
                num_adv = int(batch_size * self.config.adv_ratio)
                
                # Clean examples
                outputs_clean = self.model(images)
                loss_clean = self.criterion(outputs_clean, labels)
                
                # Adversarial examples
                if num_adv > 0:
                    loss_adv = adversarial_training_loss(
                        self.model,
                        images[:num_adv],
                        labels[:num_adv],
                        epsilon=self.config.train_epsilon,
                        alpha=self.config.train_pgd_alpha,
                        num_steps=self.config.train_pgd_steps,
                        criterion=self.criterion
                    )
                    loss = (1 - self.config.adv_ratio) * loss_clean + self.config.adv_ratio * loss_adv
                else:
                    loss = loss_clean
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                if self.adversarial_training:
                    outputs = self.model(images)  # Re-compute for clean images
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).float().mean().item() * 100
            
            # Update metrics
            metric_tracker.update({
                'train_loss': loss.item(),
                'train_acc': acc
            }, n=images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.2f}%"
            })
            
            # Log to tensorboard
            if self.global_step % self.config.log_every == 0:
                self.tb_logger.add_scalar('train/loss', loss.item(), self.global_step)
                self.tb_logger.add_scalar('train/acc', acc, self.global_step)
                self.tb_logger.add_scalar('train/lr', get_lr(self.optimizer), self.global_step)
            
            self.global_step += 1
        
        return metric_tracker.get_averages()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metric_tracker = MetricsTracker()
        
        for images, labels in tqdm(self.val_loader, desc='Validation', leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean().item() * 100
            
            # Update metrics
            metric_tracker.update({
                'val_loss': loss.item(),
                'val_acc': acc
            }, n=images.size(0))
        
        return metric_tracker.get_averages()
    
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Test the model."""
        self.model.eval()
        metric_tracker = MetricsTracker()
        
        for images, labels in tqdm(self.test_loader, desc='Testing'):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean().item() * 100
            
            # Update metrics
            metric_tracker.update({
                'test_loss': loss.item(),
                'test_acc': acc
            }, n=images.size(0))
        
        return metric_tracker.get_averages()
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info(f"Starting training: {self.config.experiment_name}")
        logger.info("=" * 80)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.2f}%")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.2f}%")
            
            # TensorBoard logging
            for key, value in val_metrics.items():
                self.tb_logger.add_scalar(f'val/{key.replace("val_", "")}', value, epoch)
            
            # WandB logging
            if self.wandb_logger:
                self.wandb_logger.log(all_metrics, step=epoch)
            
            # Save checkpoint
            is_best = val_metrics['val_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['val_acc']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            if epoch % self.config.save_every == 0 or is_best:
                self.checkpoint_manager.save(
                    step=epoch,
                    epoch=epoch,
                    metrics=all_metrics,
                    is_scheduled=False
                )
        
        # Load best model and test
        logger.info("\n" + "=" * 80)
        logger.info("Training complete! Loading best model for testing...")
        self.checkpoint_manager.load_best(device=self.device)
        
        test_metrics = self.test()
        logger.info("\nTest Results:")
        logger.info(f"  Test Loss: {test_metrics['test_loss']:.4f}")
        logger.info(f"  Test Accuracy: {test_metrics['test_acc']:.2f}%")
        logger.info("=" * 80)
        
        if self.wandb_logger:
            self.wandb_logger.log(test_metrics, step=self.config.num_epochs)
            self.wandb_logger.finish()
        
        self.tb_logger.close()
        
        return test_metrics


class AutoencoderTrainer:
    """
    Trainer for autoencoder (for OOD detection).
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Any = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Loss function
        if config.reconstruction_loss == 'mse':
            self.criterion = nn.MSELoss()
        elif config.reconstruction_loss == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss: {config.reconstruction_loss}")
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            max_keep=1,
            save_every=config.save_every,
            metric_name='val_loss',
            mode='min'
        )
        
        # Loggers
        self.tb_logger = TensorBoardLogger(
            log_dir=config.log_dir,
            enabled=config.use_tensorboard
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"AutoencoderTrainer initialized")
        logger.info(f"Model parameters: {format_params(count_parameters(model))}")
        logger.info(f"Trainable parameters: {format_params(count_parameters(model, trainable_only=True))}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for images, _ in pbar:
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructions = self.model(images)
            loss = self.criterion(reconstructions, images)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metric_tracker.update({
                'train_loss': loss.item()
            }, n=images.size(0))
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
            # Log to tensorboard
            if self.global_step % self.config.log_every == 0:
                self.tb_logger.add_scalar('train/loss', loss.item(), self.global_step)
            
            self.global_step += 1
        
        return metric_tracker.get_averages()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metric_tracker = MetricsTracker()
        
        for images, _ in tqdm(self.val_loader, desc='Validation', leave=False):
            images = images.to(self.device)
            
            # Forward pass
            reconstructions = self.model(images)
            loss = self.criterion(reconstructions, images)
            
            # Update metrics
            metric_tracker.update({
                'val_loss': loss.item()
            }, n=images.size(0))
        
        return metric_tracker.get_averages()
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info(f"Starting autoencoder training")
        logger.info("=" * 80)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.6f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.6f}")
            
            # TensorBoard logging
            self.tb_logger.add_scalar('val/loss', val_metrics['val_loss'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            if epoch % self.config.save_every == 0 or is_best:
                self.checkpoint_manager.save(
                    step=epoch,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    is_scheduled=False
                )
        
        logger.info("\n" + "=" * 80)
        logger.info("Autoencoder training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info("=" * 80)
        
        self.tb_logger.close()
        
        return {'best_val_loss': self.best_val_loss}
