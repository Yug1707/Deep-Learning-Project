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
    - Fully connected classifier with sigmoid activation
    
    Input: (batch_size, channels, height, width) = (batch_size, 10, 1, 128, 130)
    Output: (batch_size, num_classes) with sigmoid activation
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
        # Input: (batch, 1, 128, 130) -> after convs: (batch, 512, 8, 16)
        self.conv_output_size = 512 * 8 * 16
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.conv_output_size // 16,  # width after convs
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the CRNN model.
        
        Args:
            x: Input tensor of shape (batch_size, segments, channels, height, width)
               or (batch_size, channels, height, width) if segments are already batched
        
        Returns:
            Output tensor of shape (batch_size, num_classes) with sigmoid activation
        """
        # Handle both 5D (with segments) and 4D input
        if x.dim() == 5:
            batch_size, segments, channels, height, width = x.shape
            # Reshape to process all segments
            x = x.view(batch_size * segments, channels, height, width)
            has_segments = True
        else:
            batch_size = x.shape[0]
            has_segments = False
        
        # CNN feature extraction
        conv_out = self.conv_layers(x)  # (batch*segments, 512, 8, 16)
        
        # Reshape for LSTM: (batch, time_steps, features)
        batch_size_segments = conv_out.size(0)
        conv_out = conv_out.permute(0, 2, 3, 1)  # (batch*segments, 8, 16, 512)
        conv_out = conv_out.contiguous().view(batch_size_segments, 8, -1)  # (batch*segments, 8, 8192)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(conv_out)  # (batch*segments, 8, 512)
        
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]  # (batch*segments, 512)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layers
        output = self.fc_layers(lstm_out)  # (batch*segments, num_classes)
        
        # Apply sigmoid activation for multi-label classification
        output = torch.sigmoid(output)
        
        # Reshape back if we had segments
        if has_segments:
            output = output.view(batch_size, segments, self.num_classes)
            # Average across segments
            output = torch.mean(output, dim=1)  # (batch_size, num_classes)
        
        return output


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
    # Test the model
    model = create_crnn_model(num_classes=114)
    
    # Test with sample input (batch_size=4, segments=10, channels=1, height=128, width=130)
    sample_input = torch.randn(4, 10, 1, 128, 130)
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test without segments
    sample_input_2d = torch.randn(4, 1, 128, 130)
    output_2d = model(sample_input_2d)
    print(f"2D Input shape: {sample_input_2d.shape}")
    print(f"2D Output shape: {output_2d.shape}")
