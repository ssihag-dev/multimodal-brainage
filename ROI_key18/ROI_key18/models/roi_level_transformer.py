import torch
import torch.nn as nn
from typing import List, Tuple


class TransformerBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        
        return x


class ROILevelTransformer(nn.Module):
    
    def __init__(
        self,
        dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        
        # ROI-level transformer blocks
        self.roi_transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
    
    def forward(
        self,
        roi_features: torch.Tensor,  # [N_total_rois, dim]
        subject_groups: List[List[int]]  # ROI indices per subject
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        
        # ========================================
        # 1. ROI-level self-attention (per subject)
        # ========================================
        # Process each subject's ROIs independently with self-attention
        subject_roi_outputs = []
        
        for roi_indices in subject_groups:
            if len(roi_indices) == 0:
                # Empty subject (shouldn't happen)
                continue
            
            # Get ROI features for this subject
            subj_rois = roi_features[roi_indices]  # [M, dim]
            subj_rois = subj_rois.unsqueeze(0)  # [1, M, dim] for batch processing
            
            # Apply transformer blocks (self-attention among this subject's ROIs)
            for block in self.roi_transformer:
                subj_rois = block(subj_rois)
            
            subject_roi_outputs.append(subj_rois.squeeze(0))  # [M, dim]
        
        # Update all ROI features with contextualized versions
        roi_embeddings = torch.zeros_like(roi_features)
        for roi_indices, subj_rois in zip(subject_groups, subject_roi_outputs):
            roi_embeddings[roi_indices] = subj_rois
        
        # ========================================
        # 2. Subject-level aggregation
        # ========================================
        roi_embeddings_list = []
        subject_embeddings = []
        
        for roi_indices in subject_groups:
            if len(roi_indices) == 0:
                # Empty subject (edge case)
                roi_emb = torch.zeros(1, self.dim, device=roi_features.device)
            else:
                roi_emb = roi_embeddings[roi_indices]  # [num_rois, dim]
            
            roi_embeddings_list.append(roi_emb)
            
            # Subject representation: mean pooling over ROIs
            subject_feat = roi_emb.mean(dim=0)  # [dim]
            subject_embeddings.append(subject_feat)
        
        # Stack subject embeddings into batch
        subject_embeddings = torch.stack(subject_embeddings, dim=0)  # [B, dim]
        
        return roi_embeddings, roi_embeddings_list, subject_embeddings