"""
CRNN (Convolutional Recurrent Neural Network) model for multi-label audio genre classification.

This model combines CNN layers for feature extraction from spectrograms
with bidirectional LSTM layers for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    CRNN model for multi-label audio genre classification.
    
    Architecture:
    - CNN feature extractor (4 conv layers)
    - Bidirectional LSTM for temporal modeling
    - Fully connected classifier (raw logits — NO sigmoid here)
    
    Input:  (batch_size, segments, channels, height, width)  [5D]
            or (batch_size, channels, height, width)          [4D]
    Output: (batch_size, num_classes) — raw logits
            Use BCEWithLogitsLoss during training.
            Apply torch.sigmoid() explicitly for inference/metrics.
    """
    
    def __init__(self, num_classes=114, input_channels=1, hidden_size=256, num_layers=2, dropout=0.3):
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size after conv layers for the LSTM input
        # Input: (batch, 1, 128, 130) -> after convs: (batch, 512, 8, 8)
        self.conv_output_size = 512 * 8 * 8

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.conv_output_size // 8,  # features per time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers — outputs raw logits (no sigmoid)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
            # NO sigmoid here — BCEWithLogitsLoss handles it during training
        )
        
    def forward(self, x):
        """
        Forward pass through the CRNN model.
        
        Args:
            x: Input tensor of shape (batch_size, segments, channels, height, width) [5D]
               or (batch_size, channels, height, width) [4D]
        
        Returns:
            Raw logits of shape (batch_size, num_classes).
            Apply torch.sigmoid() explicitly for inference or metrics.
        """
        # Handle both 5D (with segments) and 4D input
        if x.dim() == 5:
            batch_size, segments, channels, height, width = x.shape
            x = x.view(batch_size * segments, channels, height, width)
            has_segments = True
        else:
            batch_size = x.shape[0]
            segments = 1
            has_segments = False
        
        # CNN feature extraction
        conv_out = self.conv_layers(x)  # (batch*segments, 512, H', W')
        
        # Reshape for LSTM: (batch, time_steps, features)
        batch_size_segments = conv_out.size(0)
        h = conv_out.size(2)
        conv_out = conv_out.permute(0, 2, 3, 1)  # (batch*segments, H', W', 512)
        conv_out = conv_out.contiguous().view(batch_size_segments, h, -1)  # (batch*segments, H', W'*512)
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_out)
        
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]  # (batch*segments, hidden*2)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layers → raw logits
        output = self.fc_layers(lstm_out)  # (batch*segments, num_classes)
        
        # Average across segments to get one prediction per track
        if has_segments:
            output = output.view(batch_size, segments, self.num_classes)
            output = torch.mean(output, dim=1)  # (batch_size, num_classes)
        
        return output  # raw logits — caller applies sigmoid for inference


def create_crnn_model(num_classes=114, input_channels=1, hidden_size=256, num_layers=2, dropout=0.3):
    """
    Create a CRNN model with specified parameters.
    
    Args:
        num_classes: Number of output classes (genres)
        input_channels: Number of input channels (1 for mono spectrograms)
        hidden_size: Hidden size of LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        
    Returns:
        CRNN model instance
    """
    model = CRNN(
        num_classes=num_classes,
        input_channels=input_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Initialize weights
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    model.apply(_init_weights)
    return model


if __name__ == "__main__":
    model = create_crnn_model(num_classes=114)
    
    # Test with 5D input (batch_size=4, segments=10, channels=1, height=128, width=130)
    sample_input = torch.randn(4, 10, 1, 128, 130)
    output = model(sample_input)
    print(f"5D Input shape: {sample_input.shape}")
    print(f"Output shape:   {output.shape}")   # expect (4, 114)
    print(f"Output range:   {output.min().item():.3f} to {output.max().item():.3f}  (raw logits)")
    
    # Test with 4D input
    sample_input_4d = torch.randn(4, 1, 128, 130)
    output_4d = model(sample_input_4d)
    print(f"4D Input shape: {sample_input_4d.shape}")
    print(f"Output shape:   {output_4d.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")