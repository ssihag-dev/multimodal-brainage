import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np

from models.only_encoder import MRI3DEncoder
from models.roi_pooling import ROIPool3D, ROIFeatureExtractor
from models.roi_level_transformer import ROILevelTransformer


class ROIBrainAge(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # ==================== U-Net Encoder ====================
        self.encoder = MRI3DEncoder(
            in_channels=config.in_channels,
            channels=config.encoder_channels,
            use_bn=config.use_batch_norm,
            dropout=config.dropout
        )
        
        # ==================== ROI Pooling ====================
        self.roi_pooling = ROIPool3D(output_size=config.roi_pool_size)
        
        # ==================== ROI Feature Encoder ====================
        self.roi_encoder = ROIFeatureExtractor(
            in_channels=config.roi_pool_feature_dim,
            pool_size=config.roi_pool_size,
            out_dim=config.roi_feature_dim,
            dropout=config.dropout
        )
        
        # ==================== Hierarchical Transformer ====================
        self.transformer = ROILevelTransformer(
            dim=config.transformer_dim,
            depth=config.transformer_depth,
            heads=config.transformer_heads,
            mlp_ratio=config.transformer_mlp_ratio,
            dropout=config.transformer_dropout
        )
        
        # ==================== Prediction Heads (MODIFIED) ====================
        self.roi_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.transformer_dim, 1)
        )
        
        self.subject_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.transformer_dim, 1)
        )
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary containing:
                - volume: [B, 4, D, H, W] full volumes
                - label_map: [B, D, H, W] ROI label maps
                - roi_lists: List[List[int]] ROI IDs per subject
                - age: [B] ground truth ages
        
        Returns:
            Dictionary with predictions
        """
        volumes = batch['volume']  # [B, 4, D, H, W]
        label_maps = batch['label_map']  # [B, D, H, W]
        roi_lists = batch['roi_lists']  # List of List[roi_id]
        
        B = volumes.shape[0]
        
        # ========================================
        # 1. U-Net Encoding
        # ========================================
        skip_features, bottleneck = self.encoder(volumes)
        
        # Select feature map at desired level for ROI pooling
        feature_map = skip_features[self.config.roi_pool_level]  # [B, C, D', H', W']
        
        # ========================================
        # 2. ROI Pooling and Feature Extraction
        # ========================================
        all_roi_features = []
        subject_groups = []  # ONE entry per SUBJECT
        
        roi_global_idx = 0
        
        # ========================================
        # CRITICAL: Loop over SUBJECTS, not ROIs!
        # ========================================
        for b_idx in range(B):  # ← Loop over BATCH (subjects)
            label_map = label_maps[b_idx]  # [D, H, W]
            roi_ids = roi_lists[b_idx]  # List of ROI IDs for this subject
            
            subject_roi_features = []
            subject_roi_indices = []
            
            # Process each ROI for THIS subject
            for roi_id in roi_ids:
                # Extract ROI mask
                roi_mask = (label_map == roi_id).float()  # [D, H, W]
                
                # Check if ROI has enough voxels
                if roi_mask.sum() < self.config.min_roi_voxels:
                    continue
                
                # ROI pooling
                roi_pooled = self.roi_pooling(
                    feature_map[b_idx:b_idx+1],  # [1, C, D', H', W']
                    roi_mask  # [D, H, W]
                )  # [1, C, pool_d, pool_h, pool_w]
                
                # Extract ROI features
                roi_feat = self.roi_encoder(roi_pooled)  # [1, 512]
                roi_feat = roi_feat.squeeze(0)  # [512]
                
                subject_roi_features.append(roi_feat)
                subject_roi_indices.append(roi_global_idx)
                
                roi_global_idx += 1
            
            # ========================================
            # CRITICAL: Append ONCE per subject, AFTER ROI loop
            # ========================================
            if len(subject_roi_features) > 0:
                subject_roi_stack = torch.stack(subject_roi_features, dim=0)  # [N_roi, 512]
                all_roi_features.append(subject_roi_stack)
                subject_groups.append(subject_roi_indices)
            else:
                # No valid ROIs for this subject (edge case)
                dummy_feat = torch.zeros(1, self.config.roi_feature_dim, device=volumes.device)
                all_roi_features.append(dummy_feat)
                subject_groups.append([roi_global_idx])
                roi_global_idx += 1
        
        # Validation
        assert len(subject_groups) == B, \
            f"Subject groups mismatch: {len(subject_groups)} != {B}"
        
        # Concatenate all ROI features
        all_roi_features = torch.cat(all_roi_features, dim=0)  # [Total_ROIs, 512]
                
        # ========================================
        # 3. Hierarchical Transformer
        # ========================================
        roi_out, roi_out_list, subject_out = self.transformer(
            all_roi_features,
            subject_groups
        )
                
        # ========================================
        # 4. Predictions
        # ========================================
        # ROI-level predictions (each ROI predicts subject age)
        roi_ages = []
        start_idx = 0
        for roi_indices in subject_groups:
            num_rois = len(roi_indices)
            subj_roi_out = roi_out[start_idx:start_idx+num_rois]
            roi_age = self.roi_head(subj_roi_out).squeeze(-1)  # [num_rois]
            roi_ages.append(roi_age)
            start_idx += num_rois
        
        # Subject-level predictions
        subject_ages = self.subject_head(subject_out).squeeze(-1)  # [B]
                
        return {
            'roi_ages': roi_ages,            # List of [num_rois] per subject
            'subject_ages': subject_ages,    # [B]
            'roi_features': roi_out,         # For analysis
            'subject_features': subject_out  # For analysis
        }