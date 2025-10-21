"""
Improved U-Net Architecture for BraTS2020 Brain Tumor Segmentation

This module implements an enhanced U-Net with:
1. Residual Blocks: Enable deeper networks and more stable training
2. Attention Gates: Focus on relevant tumor regions during upsampling
3. Dropout: Prevent overfitting on medical imaging data

The architecture is specifically designed for multi-modal MRI brain tumor segmentation
with 4 input channels (FLAIR, T1, T1CE, T2) and 4 output classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection.
    
    Implements a residual connection (skip connection) that adds the input to the output.
    This helps with:
    - Gradient flow during backpropagation (solves vanishing gradient problem)
    - Training stability in deeper networks
    - Better feature learning by allowing identity mapping
    
    Architecture:
        Input → Conv → BatchNorm → ReLU → Conv → BatchNorm → (+) → ReLU → Output
                                                               ↑
                                                           Identity/1x1 Conv
    
    If input and output channels differ, a 1x1 convolution is used to match dimensions.
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize Residual Block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(ResidualBlock, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection: identity or 1x1 conv if channels differ
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        identity = self.skip(x)  # Save input for skip connection
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out = out + identity
        out = self.relu(out)
        
        return out


class AttentionGate(nn.Module):
    """
    Attention Gate for focusing on relevant features.
    
    Attention gates help the model focus on salient regions (tumor areas) while
    suppressing irrelevant features (healthy tissue). This is especially important
    in medical imaging where the region of interest is small compared to the background.
    
    The attention mechanism works by:
    1. Taking features from encoder (skip connection) and decoder (upsampled)
    2. Computing attention coefficients that highlight important regions
    3. Multiplying input features by attention map to emphasize relevant areas
    
    This improves segmentation accuracy, especially for small tumor sub-regions.
    
    Reference: Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
    """
    
    def __init__(self, gate_channels, input_channels):
        """
        Initialize Attention Gate.
        
        Args:
            gate_channels (int): Number of channels in gating signal (from decoder)
            input_channels (int): Number of channels in input features (from encoder)
        """
        super(AttentionGate, self).__init__()
        
        # Intermediate channel dimension
        inter_channels = input_channels // 2
        if inter_channels == 0:
            inter_channels = 1
        
        # Transform gating signal (from decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Transform input features (from encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(input_channels, inter_channels, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, x):
        """
        Forward pass of attention gate.
        
        Args:
            gate (torch.Tensor): Gating signal from decoder (B, gate_channels, H, W)
            x (torch.Tensor): Input features from encoder (B, input_channels, H, W)
        
        Returns:
            torch.Tensor: Attention-weighted features (B, input_channels, H, W)
        """
        # Transform both inputs to intermediate dimension
        g = self.W_g(gate)
        x_trans = self.W_x(x)
        
        # Add and apply ReLU
        attention = self.relu(g + x_trans)
        
        # Compute attention coefficients (0 to 1)
        attention = self.psi(attention)
        
        # Multiply input features by attention map
        output = x * attention
        
        return output


class ImprovedUNet(nn.Module):
    """
    Improved U-Net for BraTS2020 Brain Tumor Segmentation.
    
    This enhanced U-Net architecture combines:
    1. Residual Blocks: For stable training and better gradient flow
    2. Attention Gates: To focus on tumor regions during reconstruction
    3. Dropout: To prevent overfitting on limited medical imaging data
    
    Architecture Overview:
    - Input: 4 channels (FLAIR, T1, T1CE, T2 MRI modalities)
    - Encoder: 5 levels with increasing channels (64→128→256→512→1024)
    - Decoder: 4 levels with attention gates and skip connections
    - Output: 4 classes (Background, NCR/NET, Edema, Enhancing Tumor)
    
    The model uses:
    - MaxPooling for downsampling in encoder
    - Transposed Convolution for upsampling in decoder
    - Attention gates before concatenating skip connections
    - Dropout after each encoder block for regularization
    """
    
    def __init__(self, in_channels=4, num_classes=4):
        """
        Initialize Improved U-Net.
        
        Args:
            in_channels (int): Number of input channels (default: 4 for BraTS modalities)
            num_classes (int): Number of output classes (default: 4 for BraTS classes)
        """
        super(ImprovedUNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # =============================================================================
        # ENCODER PATH (Contracting Path)
        # =============================================================================
        
        # Initial block: 4 → 64
        self.initial = ResidualBlock(in_channels, 64)
        self.dropout_initial = nn.Dropout2d(p=0.3)
        
        # Downsampling and encoding
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = ResidualBlock(64, 128)
        self.dropout1 = nn.Dropout2d(p=0.3)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = ResidualBlock(128, 256)
        self.dropout2 = nn.Dropout2d(p=0.3)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = ResidualBlock(256, 512)
        self.dropout3 = nn.Dropout2d(p=0.3)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = ResidualBlock(512, 1024)  # Bottleneck
        self.dropout4 = nn.Dropout2d(p=0.3)
        
        # =============================================================================
        # DECODER PATH (Expanding Path)
        # =============================================================================
        
        # Up1: 1024 → 512
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attention1 = AttentionGate(gate_channels=512, input_channels=512)
        self.up_block1 = ResidualBlock(1024, 512)  # 512 (from skip) + 512 (upsampled) = 1024
        
        # Up2: 512 → 256
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attention2 = AttentionGate(gate_channels=256, input_channels=256)
        self.up_block2 = ResidualBlock(512, 256)  # 256 + 256 = 512
        
        # Up3: 256 → 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attention3 = AttentionGate(gate_channels=128, input_channels=128)
        self.up_block3 = ResidualBlock(256, 128)  # 128 + 128 = 256
        
        # Up4: 128 → 64
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attention4 = AttentionGate(gate_channels=64, input_channels=64)
        self.up_block4 = ResidualBlock(128, 64)  # 64 + 64 = 128
        
        # =============================================================================
        # FINAL OUTPUT LAYER
        # =============================================================================
        
        # Final 1x1 convolution to get class predictions
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass of Improved U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, 4) or (B, 4, H, W)
                             where 4 channels are [FLAIR, T1, T1CE, T2]
        
        Returns:
            torch.Tensor: Output logits of shape (B, 4, H, W)
                         where 4 classes are [Background, NCR/NET, Edema, Enhancing]
        """
        # Handle input format from dataset (B, H, W, 4) → (B, 4, H, W)
        if x.dim() == 4 and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2)
        
        # =============================================================================
        # ENCODER PATH - Save skip connections
        # =============================================================================
        
        # Initial block
        enc1 = self.initial(x)  # 64 channels
        enc1 = self.dropout_initial(enc1)
        
        # Down1
        x1 = self.pool1(enc1)
        enc2 = self.down1(x1)  # 128 channels
        enc2 = self.dropout1(enc2)
        
        # Down2
        x2 = self.pool2(enc2)
        enc3 = self.down2(x2)  # 256 channels
        enc3 = self.dropout2(enc3)
        
        # Down3
        x3 = self.pool3(enc3)
        enc4 = self.down3(x3)  # 512 channels
        enc4 = self.dropout3(enc4)
        
        # Down4 (Bottleneck)
        x4 = self.pool4(enc4)
        bottleneck = self.down4(x4)  # 1024 channels
        bottleneck = self.dropout4(bottleneck)
        
        # =============================================================================
        # DECODER PATH - Upsample and concatenate with attention-weighted skip connections
        # =============================================================================
        
        # Up1
        dec1 = self.up1(bottleneck)  # Upsample to 512 channels
        enc4_att = self.attention1(gate=dec1, x=enc4)  # Apply attention to skip connection
        dec1 = torch.cat([enc4_att, dec1], dim=1)  # Concatenate: 512 + 512 = 1024
        dec1 = self.up_block1(dec1)  # Process to 512 channels
        
        # Up2
        dec2 = self.up2(dec1)  # Upsample to 256 channels
        enc3_att = self.attention2(gate=dec2, x=enc3)  # Apply attention
        dec2 = torch.cat([enc3_att, dec2], dim=1)  # Concatenate: 256 + 256 = 512
        dec2 = self.up_block2(dec2)  # Process to 256 channels
        
        # Up3
        dec3 = self.up3(dec2)  # Upsample to 128 channels
        enc2_att = self.attention3(gate=dec3, x=enc2)  # Apply attention
        dec3 = torch.cat([enc2_att, dec3], dim=1)  # Concatenate: 128 + 128 = 256
        dec3 = self.up_block3(dec3)  # Process to 128 channels
        
        # Up4
        dec4 = self.up4(dec3)  # Upsample to 64 channels
        enc1_att = self.attention4(gate=dec4, x=enc1)  # Apply attention
        dec4 = torch.cat([enc1_att, dec4], dim=1)  # Concatenate: 64 + 64 = 128
        dec4 = self.up_block4(dec4)  # Process to 64 channels
        
        # =============================================================================
        # FINAL OUTPUT
        # =============================================================================
        
        # Final 1x1 convolution to get class logits
        output = self.final_conv(dec4)  # Output: (B, 4, H, W)
        
        return output
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# MODEL FACTORY FUNCTION
# =============================================================================

def create_model(in_channels=None, num_classes=None, device=None):
    """
    Factory function to create an Improved U-Net model.
    
    This function creates a model and provides useful information about the model architecture.
    
    Args:
        in_channels (int, optional): Number of input channels. Defaults to 4
        num_classes (int, optional): Number of output classes. Defaults to 4
        device (torch.device, optional): Device to place model on. Defaults to cuda/cpu
    
    Returns:
        ImprovedUNet: Initialized model ready for training
    
    Example:
        >>> from unet_model import create_model
        >>> model = create_model()
        >>> print(f"Model has {model.count_parameters():,} parameters")
    """
    # Use default values if not provided
    if in_channels is None:
        in_channels = 4
    
    if num_classes is None:
        num_classes = 4
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ImprovedUNet(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)
    
    # Print model information
    print("\n" + "="*80)
    print("Improved U-Net Model Created")
    print("="*80)
    print(f"  Architecture: U-Net with Residual Blocks + Attention Gates")
    print(f"  Input Channels: {in_channels} (BraTS modalities: FLAIR, T1, T1CE, T2)")
    print(f"  Output Classes: {num_classes} (Background, NCR/NET, Edema, Enhancing)")
    print(f"  Trainable Parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    print("="*80 + "\n")
    
    return model


# =============================================================================
# TESTING - Verify model architecture
# =============================================================================

if __name__ == "__main__":
    print("\nTesting Improved U-Net Architecture...")
    print("="*80)
    
    # Create dummy input (batch=2, channels=4, height=256, width=256)
    batch_size = 2
    in_channels = 4
    num_classes = 4
    height, width = 256, 256
    
    dummy_input = torch.randn(batch_size, in_channels, height, width)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Create model
    model = create_model()
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {num_classes}, {height}, {width})")
    
    # Verify output shape
    assert output.shape == (batch_size, num_classes, height, width), \
        "Output shape mismatch!"
    
    print("\n" + "="*80)
    print("Model architecture test passed! ✓")
    print("="*80 + "\n")

