import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class ROIPool3D(nn.Module):
    """
    3D ROI Pooling.
    
    Given a feature map and ROI masks, extract fixed-size features for each ROI.
    Similar to RoI Align in 2D object detection.
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int, int] = (4, 4, 4),
    ):
        super().__init__()
        self.output_size = output_size
    
    def forward(
        self,
        features: torch.Tensor,  # [B, C, D', H', W'] feature map
        roi_mask: torch.Tensor,  # [D, H, W] binary mask in ORIGINAL resolution
    ) -> torch.Tensor:
        """
        Extract ROI features.
        
        Args:
            features: [B, C, D', H', W'] feature map (downsampled)
            roi_mask: [D, H, W] binary ROI mask (original resolution)
        
        Returns:
            roi_features: [B, C, out_d, out_h, out_w]
        """
        B, C, D_feat, H_feat, W_feat = features.shape
        D_orig, H_orig, W_orig = roi_mask.shape
        
        # ========================================
        # 1. Downsample ROI mask to match feature map
        # ========================================
        roi_mask_down = F.interpolate(
            roi_mask.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, D, H, W]
            size=(D_feat, H_feat, W_feat),
            mode='nearest'
        ).squeeze(0).squeeze(0)  # [D', H', W']
        
        # ========================================
        # 2. Find ROI bounding box in feature space
        # ========================================
        nonzero = torch.nonzero(roi_mask_down > 0.5)  # [N, 3]
        
        if nonzero.shape[0] == 0:
            # Empty ROI - return zeros
            return torch.zeros(
                B, C, *self.output_size,
                dtype=features.dtype,
                device=features.device
            )
        
        # Get bounding box
        d_min, h_min, w_min = nonzero.min(dim=0).values
        d_max, h_max, w_max = nonzero.max(dim=0).values
        
        # Add small margin
        d_min = max(0, d_min - 1)
        h_min = max(0, h_min - 1)
        w_min = max(0, w_min - 1)
        d_max = min(D_feat - 1, d_max + 1)
        h_max = min(H_feat - 1, h_max + 1)
        w_max = min(W_feat - 1, w_max + 1)
        
        # ========================================
        # 3. Extract ROI region from features
        # ========================================
        roi_features = features[
            :, :,
            d_min:d_max+1,
            h_min:h_max+1,
            w_min:w_max+1
        ]  # [B, C, d_roi, h_roi, w_roi]
        
        # ========================================
        # 4. Resize to fixed output size
        # ========================================
        roi_features = F.interpolate(
            roi_features,
            size=self.output_size,
            mode='trilinear',
            align_corners=False
        )  # [B, C, out_d, out_h, out_w]
        
        # ========================================
        # 5. Apply mask (zero out non-ROI regions)
        # ========================================
        # Resize mask to output size
        roi_mask_resized = F.interpolate(
            roi_mask_down[d_min:d_max+1, h_min:h_max+1, w_min:w_max+1].unsqueeze(0).unsqueeze(0),
            size=self.output_size,
            mode='nearest'
        ).squeeze(0).squeeze(0)  # [out_d, out_h, out_w]
        
        # Apply mask
        roi_features = roi_features * roi_mask_resized.unsqueeze(0).unsqueeze(0)
        
        return roi_features


class ROIFeatureExtractor(nn.Module):
    """
    Extract fixed-dimension features from pooled ROI.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        pool_size: Tuple[int, int, int] = (4, 4, 4),
        out_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CNN to process pooled ROI
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # FC
        self.fc = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, d, h, w] pooled ROI features
        
        Returns:
            features: [B, out_dim] ROI embedding
        """
        x = self.conv(x)  # [B, 512, d, h, w]
        x = self.global_pool(x)  # [B, 512, 1, 1, 1]
        x = x.flatten(1)  # [B, 512]
        x = self.fc(x)  # [B, out_dim]
        
        return x