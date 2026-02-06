import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_for_all.checkpoint import CheckpointManager
from utils_for_all.logger import TensorBoardLogger
from utils_for_all.metrics import MetricsTracker
from utils_for_all.misc import count_parameters, format_params, get_lr

logger = logging.getLogger(__name__)

class SentimentTrainer:
    """
    Trainer for sentiment classification.
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
        binary: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.binary = binary
        
        # Loss function
        if binary:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            max_keep=2,
            save_every=1,  # Save every epoch
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
        self.early_stop_counter = 0
        
        logger.info(f"Trainer initialized for {config.experiment_name}")
        logger.info(f"Model parameters: {format_params(count_parameters(model))}")
        logger.info(f"Trainable parameters: {format_params(count_parameters(model, trainable_only=True))}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            if self.binary:
                loss = self.criterion(logits, labels.float())
                # Compute accuracy for binary
                preds = (torch.sigmoid(logits) > 0.5).long()
                acc = (preds == labels).float().mean().item() * 100
            else:
                loss = self.criterion(logits, labels)
                # Compute accuracy for multi-class
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean().item() * 100
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metric_tracker.update({
                'train_loss': loss.item(),
                'train_acc': acc
            }, n=input_ids.size(0))
            
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
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            if self.binary:
                loss = self.criterion(logits, labels.float())
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                loss = self.criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            # Update metrics
            acc = (preds == labels).float().mean().item() * 100
            metric_tracker.update({
                'val_loss': loss.item(),
                'val_acc': acc
            }, n=input_ids.size(0))
        
        # Compute additional metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='binary')
        precision = precision_score(all_labels.numpy(), all_preds.numpy(), average='binary')
        recall = recall_score(all_labels.numpy(), all_preds.numpy(), average='binary')
        
        metrics = metric_tracker.get_averages()
        metrics['val_f1'] = f1
        metrics['val_precision'] = precision
        metrics['val_recall'] = recall
        
        return metrics
    
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Test the model."""
        self.model.eval()
        metric_tracker = MetricsTracker()
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.test_loader, desc='Testing'):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            if self.binary:
                loss = self.criterion(logits, labels.float())
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                loss = self.criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            # Update metrics
            acc = (preds == labels).float().mean().item() * 100
            metric_tracker.update({
                'test_loss': loss.item(),
                'test_acc': acc
            }, n=input_ids.size(0))
        
        # Compute additional metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='binary')
        precision = precision_score(all_labels.numpy(), all_preds.numpy(), average='binary')
        recall = recall_score(all_labels.numpy(), all_preds.numpy(), average='binary')
        
        metrics = metric_tracker.get_averages()
        metrics['test_f1'] = f1
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(all_labels.numpy(), all_preds.numpy(), 
                                                 target_names=['Negative', 'Positive']))
        
        return metrics
    
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
                self.scheduler.step()
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.2f}%")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.2f}%")
            logger.info(f"  Val F1: {val_metrics['val_f1']:.4f}, Precision: {val_metrics['val_precision']:.4f}, Recall: {val_metrics['val_recall']:.4f}")
            
            # TensorBoard logging
            for key, value in val_metrics.items():
                self.tb_logger.add_scalar(f'val/{key.replace("val_", "")}', value, epoch)

            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            self.checkpoint_manager.save(
                step=epoch,
                epoch=epoch,
                metrics=all_metrics,
                is_scheduled=False
            )
            
            # Early stopping
            if self.config.early_stopping:
                if self.early_stop_counter >= self.config.early_stopping_patience:
                    logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        # Load best model and test
        logger.info("\n" + "=" * 80)
        logger.info("Training complete! Loading best model for testing...")
        self.checkpoint_manager.load_best(device=self.device)
        
        test_metrics = self.test()
        logger.info("\nTest Results:")
        logger.info(f"  Test Loss: {test_metrics['test_loss']:.4f}")
        logger.info(f"  Test Accuracy: {test_metrics['test_acc']:.2f}%")
        logger.info(f"  Test F1: {test_metrics['test_f1']:.4f}")
        logger.info(f"  Test Precision: {test_metrics['test_precision']:.4f}")
        logger.info(f"  Test Recall: {test_metrics['test_recall']:.4f}")
        logger.info("=" * 80)
        
        self.tb_logger.close()
        
        return test_metrics
