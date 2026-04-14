"""
Main training script for multi-label audio genre classification.

This script integrates the CRNN model with the existing data pipeline
from audio_data_pipeline.ipynb and handles the complete training process.
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing data pipeline components
from audio_data_pipeline import CFG, FMADataset, train_loader, val_loader, DEVICE, mlb, num_genres
from training.trainer import Trainer, create_default_config


def load_data_pipeline():
    """
    Load the existing data pipeline from audio_data_pipeline.ipynb.
    
    Returns:
        Tuple of (train_loader, val_loader, num_classes, device)
    """
    print("Loading data pipeline...")
    
    # The data pipeline should already be executed from audio_data_pipeline.ipynb
    # We're importing the already created datasets and loaders
    
    # Ensure the datasets are properly configured
    if 'train_loader' not in globals() or 'val_loader' not in globals():
        raise RuntimeError(
            "Data pipeline not loaded. Please run audio_data_pipeline.ipynb first "
            "to create the datasets and data loaders."
        )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Number of genres: {num_genres}")
    print(f"Device: {DEVICE}")
    
    return train_loader, val_loader, num_genres, DEVICE


def create_training_config(args) -> dict:
    """
    Create training configuration from arguments and defaults.
    
    Args:
        args: Command line arguments
    
    Returns:
        Training configuration dictionary
    """
    config = create_default_config()
    
    # Override with command line arguments
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.hidden_size:
        config['hidden_size'] = args.hidden_size
    if args.num_layers:
        config['num_layers'] = args.num_layers
    if args.dropout:
        config['dropout'] = args.dropout
    if args.weight_decay:
        config['weight_decay'] = args.weight_decay
    if args.mixup_alpha:
        config['mixup_alpha'] = args.mixup_alpha
    if args.threshold:
        config['threshold'] = args.threshold
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.log_dir:
        config['log_dir'] = args.log_dir
    
    # Update device
    config['device'] = str(DEVICE)
    config['num_classes'] = num_genres
    
    return config


def save_config(config: dict, save_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CRNN model for multi-label audio genre classification')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--hidden-size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    # Path parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Other parameters
    parser.add_argument('--no-mixup', action='store_true', help='Disable mixup augmentation')
    parser.add_argument('--no-amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--save-config', type=str, default=None, help='Save configuration to file')
    
    args = parser.parse_args()
    
    # Load data pipeline
    try:
        train_loader_local, val_loader_local, num_classes, device = load_data_pipeline()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please run the audio_data_pipeline.ipynb notebook first to create the datasets.")
        return
    
    # Create training configuration
    config = create_training_config(args)
    
    # Update config based on arguments
    if args.no_mixup:
        config['use_mixup'] = False
    if args.no_amp:
        config['use_amp'] = False
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, args.save_config)
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Print configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("="*50 + "\n")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
            print(f"Resumed training from {args.resume}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")
    
    # Start training
    try:
        trainer.train(train_loader_local, val_loader_local, config['num_epochs'])
        print("\nTraining completed successfully!")
        
        # Print final results
        best_f1, best_epoch = trainer.metrics_tracker.get_best_metric('f1_macro', 'val', 'max')
        print(f"Best validation F1 (macro): {best_f1:.4f} at epoch {best_epoch + 1}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save current state
        trainer.save_checkpoint(trainer.current_epoch, 0.0, False)
        print(f"Current state saved to {config['checkpoint_dir']}/latest.pth")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
