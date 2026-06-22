import torch
import torch.nn as nn
from typing import Tuple, List


class DoubleConv(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = []
        
        # First conv
        layers.append(nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second conv
        layers.append(nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)


class MRI3DEncoder(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 4,
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        use_bn: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_levels = len(channels)
        
        # Encoder levels
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        current_channels = in_channels
        
        for i, out_ch in enumerate(channels):
            # Double conv
            self.encoders.append(
                DoubleConv(
                    current_channels,
                    out_ch,
                    use_bn=use_bn,
                    dropout=dropout if i == len(channels) - 1 else 0.0
                )
            )
            
            # Max pooling (except for last level)
            if i < len(channels) - 1:
                self.pools.append(nn.MaxPool3d(2, stride=2))
            
            current_channels = out_ch
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        skip_features = []
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            
            if i < len(self.pools):
                skip_features.append(x)
                x = self.pools[i](x)
        
        bottleneck = x
        
        return skip_features, bottleneck