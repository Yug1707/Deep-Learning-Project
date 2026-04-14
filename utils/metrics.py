"""
Evaluation metrics for multi-label audio genre classification.

This module implements various metrics specifically designed for
multi-label classification tasks.
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, precision_score, recall_score
from typing import Dict, List, Tuple, Optional


class MultiLabelMetrics:
    """
    Collection of metrics for multi-label classification evaluation.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Threshold for converting probabilities to binary predictions
        """
        self.threshold = threshold
    
    def _threshold_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert probability predictions to binary predictions.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
        
        Returns:
            Binary predictions of shape (batch_size, num_classes)
        """
        return (predictions >= self.threshold).float()
    
    def calculate_f1_score(self, predictions: torch.Tensor, targets: torch.Tensor, 
                          average: str = 'macro') -> float:
        """
        Calculate F1 score for multi-label classification.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
            average: Averaging method ('macro', 'micro', 'weighted', 'samples')
        
        Returns:
            F1 score
        """
        pred_binary = self._threshold_predictions(predictions)
        
        # Convert to numpy for sklearn
        pred_np = pred_binary.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        return f1_score(target_np, pred_np, average=average, zero_division=0)
    
    def calculate_precision(self, predictions: torch.Tensor, targets: torch.Tensor,
                           average: str = 'macro') -> float:
        """
        Calculate precision for multi-label classification.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
            average: Averaging method ('macro', 'micro', 'weighted', 'samples')
        
        Returns:
            Precision score
        """
        pred_binary = self._threshold_predictions(predictions)
        
        pred_np = pred_binary.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        return precision_score(target_np, pred_np, average=average, zero_division=0)
    
    def calculate_recall(self, predictions: torch.Tensor, targets: torch.Tensor,
                        average: str = 'macro') -> float:
        """
        Calculate recall for multi-label classification.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
            average: Averaging method ('macro', 'micro', 'weighted', 'samples')
        
        Returns:
            Recall score
        """
        pred_binary = self._threshold_predictions(predictions)
        
        pred_np = pred_binary.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        return recall_score(target_np, pred_np, average=average, zero_division=0)
    
    def calculate_roc_auc(self, predictions: torch.Tensor, targets: torch.Tensor,
                         average: str = 'macro') -> float:
        """
        Calculate ROC AUC score for multi-label classification.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
            average: Averaging method ('macro', 'micro')
        
        Returns:
            ROC AUC score
        """
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        try:
            return roc_auc_score(target_np, pred_np, average=average)
        except ValueError:
            # Handle case where only one class is present
            return 0.0
    
    def calculate_hamming_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Hamming loss for multi-label classification.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
        
        Returns:
            Hamming loss
        """
        pred_binary = self._threshold_predictions(predictions)
        
        pred_np = pred_binary.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        return hamming_loss(target_np, pred_np)
    
    def calculate_subset_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate subset accuracy (exact match) for multi-label classification.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
        
        Returns:
            Subset accuracy
        """
        pred_binary = self._threshold_predictions(predictions)
        
        # Check if all labels match exactly
        matches = torch.all(pred_binary == targets, dim=1)
        return matches.float().mean().item()
    
    def calculate_precision_at_k(self, predictions: torch.Tensor, targets: torch.Tensor,
                                k: int = 5) -> float:
        """
        Calculate precision@k for multi-label classification.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
            k: Number of top predictions to consider
        
        Returns:
            Precision@k score
        """
        batch_size = predictions.size(0)
        
        # Get top-k predictions
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # Check how many of the top-k predictions are actually present in targets
        correct = 0
        total = 0
        
        for i in range(batch_size):
            pred_labels = top_k_indices[i]
            true_labels = torch.where(targets[i] == 1)[0]
            
            # Count correct predictions
            intersection = len(set(pred_labels.tolist()) & set(true_labels.tolist()))
            correct += intersection
            total += k
        
        return correct / total if total > 0 else 0.0
    
    def calculate_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            predictions: Probability predictions of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
        
        Returns:
            Dictionary containing all metrics
        """
        targets = (targets >= 0.5).float()
        metrics = {
            'f1_macro': self.calculate_f1_score(predictions, targets, 'macro'),
            'f1_micro': self.calculate_f1_score(predictions, targets, 'micro'),
            'precision_macro': self.calculate_precision(predictions, targets, 'macro'),
            'precision_micro': self.calculate_precision(predictions, targets, 'micro'),
            'recall_macro': self.calculate_recall(predictions, targets, 'macro'),
            'recall_micro': self.calculate_recall(predictions, targets, 'micro'),
            'roc_auc_macro': self.calculate_roc_auc(predictions, targets, 'macro'),
            'roc_auc_micro': self.calculate_roc_auc(predictions, targets, 'micro'),
            'hamming_loss': self.calculate_hamming_loss(predictions, targets),
            'subset_accuracy': self.calculate_subset_accuracy(predictions, targets),
            'precision_at_5': self.calculate_precision_at_k(predictions, targets, k=5),
            'precision_at_10': self.calculate_precision_at_k(predictions, targets, k=10),
        }
        
        return metrics


class AverageMeter:
    """
    Utility class to track average values during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update the average meter.
        
        Args:
            val: New value to add
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """
    Track metrics during training and validation.
    """
    
    def __init__(self):
        self.metrics_history = {
            'train': {},
            'val': {}
        }
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int):
        """Update current epoch number."""
        self.current_epoch = epoch
    
    def log_metrics(self, metrics: Dict[str, float], phase: str = 'train'):
        """
        Log metrics for current epoch.
        
        Args:
            metrics: Dictionary of metric names and values
            phase: 'train' or 'val'
        """
        if phase not in self.metrics_history:
            self.metrics_history[phase] = {}
        
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.metrics_history[phase]:
                self.metrics_history[phase][metric_name] = []
            
            self.metrics_history[phase][metric_name].append(metric_value)
    
    def get_best_metric(self, metric_name: str, phase: str = 'val', mode: str = 'max') -> Tuple[float, int]:
        """
        Get best value and epoch for a specific metric.
        
        Args:
            metric_name: Name of the metric
            phase: 'train' or 'val'
            mode: 'max' or 'min' to determine best value
        
        Returns:
            Tuple of (best_value, best_epoch)
        """
        if metric_name not in self.metrics_history[phase]:
            return 0.0, 0
        
        values = self.metrics_history[phase][metric_name]
        if mode == 'max':
            best_value = max(values)
        else:
            best_value = min(values)
        
        best_epoch = values.index(best_value)
        return best_value, best_epoch
    
    def get_metrics_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        summary = {}
        
        for phase in ['train', 'val']:
            summary[phase] = {}
            for metric_name, values in self.metrics_history[phase].items():
                if values:
                    summary[phase][metric_name] = {
                        'current': values[-1],
                        'best': max(values) if 'loss' not in metric_name else min(values),
                        'best_epoch': values.index(max(values)) if 'loss' not in metric_name else values.index(min(values))
                    }
        
        return summary


if __name__ == "__main__":
    # Test metrics
    metrics_calculator = MultiLabelMetrics(threshold=0.5)
    
    # Create sample predictions and targets
    batch_size, num_classes = 32, 114
    predictions = torch.rand(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Calculate all metrics
    all_metrics = metrics_calculator.calculate_all_metrics(predictions, targets)
    
    print("Metrics Test Results:")
    for metric_name, value in all_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.log_metrics(all_metrics, phase='train')
    tracker.log_metrics(all_metrics, phase='val')
    
    summary = tracker.get_metrics_summary()
    print("\nMetrics Summary:")
    print(summary)
