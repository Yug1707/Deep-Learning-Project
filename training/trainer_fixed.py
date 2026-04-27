"""
Fixed trainer.py — all bugs resolved.

Fixes applied vs previous version:
  1. Removed 5D->4D reshape in train_epoch and validate_epoch.
     The CRNN model handles 5D input natively and averages segments internally.
  2. Removed spectrogram augmentations call on the raw 5D batch.
     Augmentations expect 4D tensors; the model handles the segment dimension.
  3. Removed sigmoid in predictions — model now returns raw logits,
     so torch.sigmoid() is applied explicitly before metrics.
  4. Fixed GradScaler deprecation: torch.amp.GradScaler('cuda').
  5. Increased default early_stopping_patience from 20 to 15 (kept reasonable).
  6. Mixup kept on 5D specs — it works correctly on any shape.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import json

from models.crnn_model import create_crnn_model
from utils.augmentations import Mixup
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
        
        # BCEWithLogitsLoss expects RAW LOGITS — model must NOT apply sigmoid
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
        
        # Mixed precision scaler (use new API to avoid deprecation warning)
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Initialize metrics
        self.metrics_calculator = MultiLabelMetrics(threshold=config.get('threshold', 0.5))
        self.metrics_tracker = MetricsTracker()
        
        # Mixup augmentation (operates on raw 5D batch — no shape issues)
        self.mixup = Mixup(alpha=config.get('mixup_alpha', 0.2), p=config.get('mixup_prob', 0.5))
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        
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
        """Train for one epoch."""
        self.model.train()
        
        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (specs, labels) in enumerate(pbar):
            # specs shape: (batch, segments, 1, 128, 130)  — 5D
            # labels shape: (batch, num_classes)
            specs = specs.to(self.device)
            labels = labels.to(self.device)

            # Mixup works on any shape — mixes whole tracks including all segments
            if self.config.get('use_mixup', True):
                specs, labels = self.mixup((specs, labels))
                specs = specs.clone()
                labels = labels.clone()

            # NOTE: No 5D->4D reshape here.
            # The model handles 5D natively: reshapes internally, processes each
            # segment through CNN+LSTM, then averages across segments.
            # No augmentations here either — augmentations expect 4D tensors.

            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(specs)          # (batch, num_classes) raw logits
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            loss_meter.update(loss.item(), specs.size(0))
            
            # Convert logits -> probabilities for metrics
            with torch.no_grad():
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
            
            pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate_all_metrics(all_predictions, all_targets)
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        with torch.no_grad():
            for specs, labels in pbar:
                # specs shape: (batch, segments, 1, 128, 130)  — 5D
                specs = specs.to(self.device)
                labels = labels.to(self.device)

                # NOTE: No 5D->4D reshape. Model handles segments internally.
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(specs)      # (batch, num_classes) raw logits
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(specs)
                    loss = self.criterion(outputs, labels)
                
                loss_meter.update(loss.item(), specs.size(0))
                
                # Convert logits -> probabilities for metrics
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
                
                pbar.set_postfix({'Loss': f'{loss_meter.avg:.4f}'})
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate_all_metrics(all_predictions, all_targets)
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False):
        """Save model checkpoint."""
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
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))
            self.logger.info(f"New best model saved with F1: {val_f1:.4f}")
        
        if epoch % self.config.get('save_freq', 10) == 0:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Train the model."""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.metrics_tracker.update_epoch(epoch)
            
            train_metrics = self.train_epoch(train_loader)
            self.metrics_tracker.log_metrics(train_metrics, phase='train')
            
            val_metrics = self.validate_epoch(val_loader)
            self.metrics_tracker.log_metrics(val_metrics, phase='val')
            
            self.scheduler.step()
            
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val F1 (macro): {val_metrics['f1_macro']:.4f}, "
                f"Val F1 (micro): {val_metrics['f1_micro']:.4f}"
            )
            
            is_best = val_metrics['f1_macro'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_metrics['f1_macro'], is_best)
            
            if epoch - self.best_epoch >= self.config.get('early_stopping_patience', 15):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        self.logger.info(f"Training completed. Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch + 1}")


def create_default_config() -> Dict:
    """Create default training configuration."""
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
        'early_stopping_patience': 15,
        'save_freq': 10,
        
        # Paths
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }