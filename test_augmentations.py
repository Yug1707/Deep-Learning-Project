"""
Test script to debug augmentation issues.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.augmentations import get_training_augmentations, Mixup

def test_augmentations():
    """Test augmentation pipeline with sample data."""
    print("Testing augmentations...")
    
    # Create sample 5D tensor (what our data loader produces)
    batch_size, segments, channels, height, width = 4, 10, 1, 128, 130
    sample_specs = torch.randn(batch_size, segments, channels, height, width)
    sample_labels = torch.randint(0, 2, (batch_size, 114)).float()
    
    print(f"Input specs shape: {sample_specs.shape}")
    print(f"Input labels shape: {sample_labels.shape}")
    
    # Test Mixup
    print("\nTesting Mixup...")
    mixup = Mixup(alpha=0.2, p=1.0)  # Always apply
    try:
        mixed_specs, mixed_labels = mixup((sample_specs, sample_labels))
        print(f"✓ Mixup successful: {mixed_specs.shape}")
    except Exception as e:
        print(f"✗ Mixup failed: {e}")
        return False
    
    # Test training augmentations
    print("\nTesting training augmentations...")
    train_aug = get_training_augmentations()
    try:
        aug_specs = train_aug(mixed_specs)
        print(f"✓ Training augmentations successful: {aug_specs.shape}")
    except Exception as e:
        print(f"✗ Training augmentations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All augmentation tests passed!")
    return True

if __name__ == "__main__":
    test_augmentations()
