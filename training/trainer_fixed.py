"""
Fixed version of trainer.py with syntax error corrected.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import json

from models.crnn_model import create_crnn_model
from utils.augmentations import get_training_augmentations, get_validation_augmentations, Mixup
from utils.metrics import MultiLabelMetrics, MetricsTracker, AverageMeter


class Trainer:
    """
    Trainer class for multi-label audio genre classification.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model
        self.model = create_crnn_model(
            num_classes=config.get('num_classes', 114),
            input_channels=config.get('input_channels', 1),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3)
        ).to(self.device)
        
        # Initialize loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 1e-6)
        )
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Initialize metrics
        self.metrics_calculator = MultiLabelMetrics(threshold=config.get('threshold', 0.5))
        self.metrics_tracker = MetricsTracker()
        
        # Initialize augmentations
        self.train_augmentations = get_training_augmentations()
        self.val_augmentations = get_validation_augmentations()
        self.mixup = Mixup(alpha=config.get('mixup_alpha', 0.2), p=config.get('mixup_prob', 0.5))
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        # Initialize metrics
        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (specs, labels) in enumerate(pbar):
            specs = specs.to(self.device)
            labels = labels.to(self.device)
            # Apply mixup augmentation
            if self.config.get('use_mixup', True):
                specs, labels = self.mixup((specs, labels))
                specs = specs.clone()
                labels = labels.clone()
                
            #Reshape if 5D
            if specs.dim() == 5:
                batch_size, num_segments, C, H, W = specs.shape
                specs = specs.view(batch_size * num_segments, C, H, W)
                labels = labels.unsqueeze(1).expand(-1, num_segments, -1).contiguous()
                labels = labels.view(batch_size * num_segments, -1)

            # Apply spectrogram augmentations
            specs = self.train_augmentations(specs)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(specs)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), specs.size(0))
            
            # Store predictions and targets for metrics calculation
            with torch.no_grad():
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate_all_metrics(all_predictions, all_targets)
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        # Initialize metrics
        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        with torch.no_grad():
            for specs, labels in pbar:
                specs = specs.to(self.device)
                labels = labels.to(self.device)
                # Reshape if 5D: (batch, segments, C, H, W) -> (batch*segments, C, H, W)
                if specs.dim() == 5:
                    batch_size, num_segments, C, H, W = specs.shape
                    specs = specs.view(batch_size * num_segments, C, H, W)
                    labels = labels.unsqueeze(1).expand(-1, num_segments, -1).contiguous()
                    labels = labels.view(batch_size * num_segments, -1)
                    labels = labels.to(self.device)
                
                # Apply validation augmentations (minimal)
                specs = self.val_augmentations(specs)
                
                # Forward pass
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(specs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(specs)
                    loss = self.criterion(outputs, labels)
                
                # Update metrics
                loss_meter.update(loss.item(), specs.size(0))
                
                # Store predictions and targets for metrics calculation
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss_meter.avg:.4f}'})
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate_all_metrics(all_predictions, all_targets)
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_f1: Validation F1 score
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))
            self.logger.info(f"New best model saved with F1: {val_f1:.4f}")
        
        # Save epoch checkpoint
        if epoch % self.config.get('save_freq', 10) == 0:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.metrics_tracker.update_epoch(epoch)
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.metrics_tracker.log_metrics(train_metrics, phase='train')
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.metrics_tracker.log_metrics(val_metrics, phase='val')
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val F1 (macro): {val_metrics['f1_macro']:.4f}, "
                f"Val F1 (micro): {val_metrics['f1_micro']:.4f}"
            )
            
            # Save checkpoint
            is_best = val_metrics['f1_macro'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_metrics['f1_macro'], is_best)
            
            # Early stopping
            if epoch - self.best_epoch >= self.config.get('early_stopping_patience', 20):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        self.logger.info(f"Training completed. Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch + 1}")


def create_default_config() -> Dict:
    """
    Create default training configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Model parameters
        'num_classes': 114,
        'input_channels': 1,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        
        # Training parameters
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 16,
        'num_epochs': 100,
        
        # Scheduler parameters
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-6,
        
        # Augmentation parameters
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'mixup_prob': 0.5,
        
        # Training settings
        'threshold': 0.5,
        'use_amp': True,
        'early_stopping_patience': 20,
        'save_freq': 10,
        
        # Paths
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
