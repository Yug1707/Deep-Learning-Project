"""
Inference script for trained CRNN model.

This script provides utilities for loading trained models
and making predictions on audio data.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import librosa

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.crnn_model import create_crnn_model
from utils.metrics import MultiLabelMetrics
from audio_data_pipeline import CFG, load_audio, split_segments, audio_to_logmel


class AudioGenrePredictor:
    """
    Predictor class for audio genre classification.
    """
    
    def __init__(self, model_path: str, config_path: str = None, device: str = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            device: Device to run inference on
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = checkpoint.get('config', {})
        
        # Initialize model
        self.model = create_crnn_model(
            num_classes=config.get('num_classes', 114),
            input_channels=config.get('input_channels', 1),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 2),
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store configuration
        self.config = config
        self.threshold = config.get('threshold', 0.5)
        
        # Initialize metrics calculator
        self.metrics_calculator = MultiLabelMetrics(threshold=self.threshold)
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {config.get('num_classes', 114)}")
    
    def preprocess_audio(self, audio_path: str, max_segments: int = 10) -> torch.Tensor:
        """
        Preprocess audio file for inference.
        
        Args:
            audio_path: Path to audio file
            max_segments: Maximum number of segments to use
        
        Returns:
            Preprocessed audio tensor of shape (1, max_segments, 1, height, width)
        """
        # Load audio
        audio = load_audio(Path(audio_path), CFG["sr"])
        if audio is None:
            raise ValueError(f"Could not load audio file: {audio_path}")
        
        # Split into segments
        segments = split_segments(audio, CFG["sr"])
        
        if len(segments) == 0:
            raise ValueError("Audio file too short for segmentation")
        
        # Convert segments to spectrograms
        specs = []
        for seg in segments[:max_segments]:  # Limit to max_segments
            logmel = audio_to_logmel(seg, CFG["sr"])
            spec_tensor = torch.tensor(logmel).unsqueeze(0).float()
            specs.append(spec_tensor)
        
        # Pad if necessary
        while len(specs) < max_segments:
            dummy_spec = torch.zeros(1, CFG["n_mels"], 130)
            specs.append(dummy_spec)
        
        # Stack and add batch dimension
        specs_tensor = torch.stack(specs)  # (max_segments, 1, height, width)
        specs_tensor = specs_tensor.unsqueeze(0)  # (1, max_segments, 1, height, width)
        
        return specs_tensor.to(self.device)
    
    def predict(self, audio_path: str, return_probabilities: bool = False) -> dict:
        """
        Predict genres for a single audio file.
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Whether to return raw probabilities
        
        Returns:
            Dictionary containing predictions
        """
        # Preprocess audio
        specs = self.preprocess_audio(audio_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(specs)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Convert to binary predictions
        binary_predictions = (probabilities >= self.threshold).astype(int)
        
        # Get predicted genre indices
        predicted_indices = np.where(binary_predictions == 1)[0]
        
        result = {
            'predicted_indices': predicted_indices.tolist(),
            'predicted_probabilities': probabilities[predicted_indices].tolist(),
            'all_probabilities': probabilities.tolist() if return_probabilities else None
        }
        
        return result
    
    def predict_batch(self, audio_paths: list, return_probabilities: bool = False) -> list:
        """
        Predict genres for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            return_probabilities: Whether to return raw probabilities
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path, return_probabilities)
                result['audio_path'] = audio_path
                result['success'] = True
            except Exception as e:
                result = {
                    'audio_path': audio_path,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
        
        return results


def main():
    """Example usage of the predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio genre classification inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, help='Path to model configuration')
    parser.add_argument('--audio', type=str, help='Single audio file to predict')
    parser.add_argument('--audio-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = AudioGenrePredictor(args.model, args.config, args.device)
    
    # Override threshold if provided
    if args.threshold:
        predictor.threshold = args.threshold
        predictor.metrics_calculator.threshold = args.threshold
    
    # Make predictions
    if args.audio:
        # Single file prediction
        result = predictor.predict(args.audio, return_probabilities=True)
        print(f"Predictions for {args.audio}:")
        print(f"Predicted indices: {result['predicted_indices']}")
        print(f"Predicted probabilities: {result['predicted_probabilities']}")
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
    
    elif args.audio_dir:
        # Batch prediction
        audio_dir = Path(args.audio_dir)
        audio_files = list(audio_dir.glob('*.mp3')) + list(audio_dir.glob('*.wav'))
        
        print(f"Found {len(audio_files)} audio files")
        
        results = predictor.predict_batch([str(f) for f in audio_files], return_probabilities=True)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"Successfully processed {successful}/{len(results)} files")
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
    
    else:
        print("Please provide either --audio or --audio-dir argument")


if __name__ == "__main__":
    main()
