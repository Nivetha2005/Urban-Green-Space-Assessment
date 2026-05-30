# === FILE: models/unet_model.py ===
import torch
import torch.nn as nn
import numpy as np

# Class colors for visualization (RGB format)
CLASS_COLORS = {
    0: [255, 255, 255],  # Impervious surfaces (roads)
    1: [0, 0, 255],      # Buildings
    2: [0, 255, 255],    # Low vegetation
    3: [0, 255, 0],      # Trees
    4: [255, 255, 0],    # Cars
    5: [255, 0, 0]       # Clutter
}

# Class names for legends
CLASS_NAMES = {
    0: "Impervious (Roads)",
    1: "Building",
    2: "Low Vegetation",
    3: "Tree",
    4: "Car",
    5: "Clutter"
}

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetWithDropout(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, dropout_rate=0.3):
        super().__init__()
        
        # Encoder
        self.d1 = DoubleConv(in_channels, 64)
        self.p1 = nn.MaxPool2d(2)
        
        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)
        
        self.d3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)
        
        self.d4 = DoubleConv(256, 512)
        self.p4 = nn.MaxPool2d(2)
        
        # Bottleneck with dropout
        self.b = nn.Sequential(
            DoubleConv(512, 1024),
            nn.Dropout2d(dropout_rate)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.u1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u4 = DoubleConv(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.dropout_rate = dropout_rate
    
    def forward(self, x, enable_dropout=False):
        # Encoder
        c1 = self.d1(x)
        p1 = self.p1(c1)
        
        c2 = self.d2(p1)
        p2 = self.p2(c2)
        
        c3 = self.d3(p2)
        p3 = self.p3(c3)
        
        c4 = self.d4(p3)
        p4 = self.p4(c4)
        
        # Bottleneck
        b = self.b(p4)
        
        # Decoder with skip connections
        u1 = self.up1(b)
        u1 = torch.cat([u1, c4], dim=1)
        c5 = self.u1(u1)
        
        u2 = self.up2(c5)
        u2 = torch.cat([u2, c3], dim=1)
        c6 = self.u2(u2)
        
        u3 = self.up3(c6)
        u3 = torch.cat([u3, c2], dim=1)
        c7 = self.u3(u3)
        
        u4 = self.up4(c7)
        u4 = torch.cat([u4, c1], dim=1)
        c8 = self.u4(u4)
        
        # Apply dropout at the end if requested (for MC Dropout)
        if enable_dropout:
            c8 = nn.Dropout2d(self.dropout_rate)(c8)
        
        return self.out(c8)

def mask_to_rgb(mask):
    """Convert class mask (H, W) to RGB image (H, W, 3)"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color
    
    return rgb