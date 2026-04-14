"""
Data augmentation utilities for audio spectrograms.

This module implements various augmentation techniques specifically designed
for spectrogram data in audio classification tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for Speech Recognition.
    
    Reference: https://arxiv.org/abs/1904.08779
    
    Applies time masking and frequency masking to spectrograms.
    """
    
    def __init__(self, time_masks=2, freq_masks=2, time_mask_width=10, freq_mask_width=8, p=0.5):
        self.time_masks = time_masks
        self.freq_masks = freq_masks
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.p = p
    
    def __call__(self, spec):
        if random.random() > self.p:
            return spec
        
        if spec.dim() == 3:
            return self._apply_spec_augment(spec)
        elif spec.dim() == 4:
            batch_size = spec.size(0)
            augmented_specs = []
            for i in range(batch_size):
                augmented_specs.append(self._apply_spec_augment(spec[i]))
            return torch.stack(augmented_specs)
        elif spec.dim() == 5:
            batch_size, num_segments = spec.size(0), spec.size(1)
            augmented_specs = []
            for i in range(batch_size):
                batch_augmented = []
                for j in range(num_segments):
                    batch_augmented.append(self._apply_spec_augment(spec[i, j]))
                augmented_specs.append(torch.stack(batch_augmented))
            return torch.stack(augmented_specs)
        else:
            raise ValueError(f"Expected 3D, 4D, or 5D tensor, got {spec.dim()}D")
    
    def _apply_spec_augment(self, spec):
        spec = spec.clone()
        channels, height, width = spec.shape
        
        # Apply frequency masks
        for _ in range(self.freq_masks):
            freq_mask_width = random.randint(1, self.freq_mask_width)
            freq_start = random.randint(0, max(0, height - freq_mask_width))
            spec[:, freq_start:freq_start + freq_mask_width, :] = 0
        
        # Apply time masks
        for _ in range(self.time_masks):
            time_mask_width = random.randint(1, self.time_mask_width)
            time_start = random.randint(0, max(0, width - time_mask_width))
            spec[:, :, time_start:time_start + time_mask_width] = 0
        
        return spec


class TimeShift:
    """
    Randomly shift spectrograms in time dimension.
    """
    
    def __init__(self, max_shift=10, p=0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, spec):
        if random.random() > self.p:
            return spec
        
        if spec.dim() == 3:
            return self._apply_time_shift(spec)
        elif spec.dim() == 4:
            batch_size = spec.size(0)
            augmented_specs = []
            for i in range(batch_size):
                augmented_specs.append(self._apply_time_shift(spec[i]))
            return torch.stack(augmented_specs)
        elif spec.dim() == 5:
            batch_size, num_segments = spec.size(0), spec.size(1)
            augmented_specs = []
            for i in range(batch_size):
                batch_augmented = []
                for j in range(num_segments):
                    batch_augmented.append(self._apply_time_shift(spec[i, j]))
                augmented_specs.append(torch.stack(batch_augmented))
            return torch.stack(augmented_specs)
        else:
            raise ValueError(f"Expected 3D, 4D, or 5D tensor, got {spec.dim()}D")
    
    def _apply_time_shift(self, spec):
        spec = spec.clone()
        channels, height, width = spec.shape
        
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return spec
        
        if shift > 0:
            # Shift right: copy data first to avoid overlap
            temp = spec[:, :, :-shift].clone()
            spec[:, :, shift:] = temp
            spec[:, :, :shift] = 0
        else:
            # Shift left: copy data first to avoid overlap
            temp = spec[:, :, -shift:].clone()
            spec[:, :, :shift] = temp
            spec[:, :, shift:] = 0
        
        return spec


class FrequencyShift:
    """
    Randomly shift spectrograms in frequency dimension.
    """
    
    def __init__(self, max_shift=5, p=0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, spec):
        if random.random() > self.p:
            return spec
        
        if spec.dim() == 3:
            return self._apply_freq_shift(spec)
        elif spec.dim() == 4:
            batch_size = spec.size(0)
            augmented_specs = []
            for i in range(batch_size):
                augmented_specs.append(self._apply_freq_shift(spec[i]))
            return torch.stack(augmented_specs)
        elif spec.dim() == 5:
            batch_size, num_segments = spec.size(0), spec.size(1)
            augmented_specs = []
            for i in range(batch_size):
                batch_augmented = []
                for j in range(num_segments):
                    batch_augmented.append(self._apply_freq_shift(spec[i, j]))
                augmented_specs.append(torch.stack(batch_augmented))
            return torch.stack(augmented_specs)
        else:
            raise ValueError(f"Expected 3D, 4D, or 5D tensor, got {spec.dim()}D")
    
    def _apply_freq_shift(self, spec):
        spec = spec.clone()
        channels, height, width = spec.shape
        
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return spec
        
        if shift > 0:
            # Shift down: copy data first to avoid overlap
            temp = spec[:, :-shift, :].clone()
            spec[:, shift:, :] = temp
            spec[:, :shift, :] = 0
        else:
            # Shift up: copy data first to avoid overlap
            temp = spec[:, -shift:, :].clone()
            spec[:, :shift, :] = temp
            spec[:, shift:, :] = 0
        
        return spec


class Mixup:
    """
    Mixup augmentation for multi-label classification.
    
    Reference: https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha=0.2, p=0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, batch):
        if random.random() > self.p:
            return batch
        
        specs, labels = batch
        batch_size = specs.size(0)
        
        if batch_size < 2:
            return batch
        
        # Generate mixing weights
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size)
        
        # Mix spectrograms — clone to avoid shared memory with original
        mixed_specs = (lam * specs + (1 - lam) * specs[index]).clone()
        
        # Mix labels for multi-label classification
        mixed_labels = (lam * labels + (1 - lam) * labels[index]).clone()
        
        return mixed_specs, mixed_labels


class Compose:
    """
    Compose multiple augmentations.
    """
    
    def __init__(self, augmentations):
        self.augmentations = augmentations
    
    def __call__(self, data):
        try:
            for aug in self.augmentations:
                data = aug(data)
            return data
        except ValueError as e:
            print(f"Augmentation error: {e}")
            print(f"Input tensor shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
            print(f"Augmentation pipeline: {[type(aug).__name__ for aug in self.augmentations]}")
            raise e


def get_training_augmentations():
    """
    Get standard training augmentations for spectrograms.
    
    Returns:
        Composed augmentation pipeline
    """
    return Compose([
        SpecAugment(time_masks=2, freq_masks=2, time_mask_width=10, freq_mask_width=8, p=0.8),
        TimeShift(max_shift=5, p=0.3),
        FrequencyShift(max_shift=3, p=0.3),
    ])


def get_validation_augmentations():
    """
    Get validation augmentations (minimal/no augmentation).
    
    Returns:
        Identity transform
    """
    return Compose([])


if __name__ == "__main__":
    # Test augmentations
    sample_spec = torch.randn(1, 128, 130)
    print(f"Original shape: {sample_spec.shape}")
    
    spec_aug = SpecAugment()
    time_shift = TimeShift()
    freq_shift = FrequencyShift()
    
    aug_spec = spec_aug(sample_spec)
    print(f"After SpecAugment: {aug_spec.shape}")
    
    batch_spec = torch.randn(4, 1, 128, 130)
    batch_labels = torch.randn(4, 114)
    
    train_aug = get_training_augmentations()
    aug_batch = train_aug(batch_spec)
    print(f"Batch augmentation shape: {aug_batch.shape}")
    
    mixup = Mixup(alpha=0.2, p=1.0)
    mixed_batch, mixed_labels = mixup((batch_spec, batch_labels))
    print(f"Mixed batch shape: {mixed_batch.shape}")
    print(f"Mixed labels shape: {mixed_labels.shape}")
