
import torch
import torch.nn as nn
from typing import Dict, Any, List


class ROIBrainAgeLoss(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        # Loss weights
        self.lambda_roi = config.lambda_roi
        self.lambda_subject = config.lambda_subject
        self.lambda_consistency = config.lambda_consistency
        
        # MSE loss
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        
        true_ages = batch['age']  
        subject_ages = outputs['subject_ages'] 
        roi_ages = outputs['roi_ages']  
        
        batch_size = true_ages.shape[0]
        
        # ========================================
        # 1. Subject-level MSE
        # ========================================
        subject_loss = self.mse(subject_ages, true_ages)
        
        # ========================================
        # 2. ROI-level MSE
        # ========================================
        # Each ROI should predict the subject's age
        roi_loss = 0.0
        total_rois = 0
        
        for subj_idx, subj_roi_ages in enumerate(roi_ages):
            if len(subj_roi_ages) == 0:
                continue
            
            true_age = true_ages[subj_idx]
            roi_mse = self.mse(
                subj_roi_ages,
                true_age.expand_as(subj_roi_ages)
            )
            
            roi_loss += roi_mse
            total_rois += 1
        
        if total_rois > 0:
            roi_loss = roi_loss / total_rois
        else:
            roi_loss = torch.tensor(0.0, device=true_ages.device)
        
        # ========================================
        # 3. Consistency Loss
        # ========================================
        consistency_loss = 0.0
        
        for subj_idx, subj_roi_ages in enumerate(roi_ages):
            if len(subj_roi_ages) == 0:
                continue
            
            subj_pred = subject_ages[subj_idx]
            
            consistency_mse = self.mse(
                subj_roi_ages,
                subj_pred.expand_as(subj_roi_ages)
            )
            
            consistency_loss += consistency_mse
        
        if total_rois > 0:
            consistency_loss = consistency_loss / total_rois
        else:
            consistency_loss = torch.tensor(0.0, device=true_ages.device)
        
        # ========================================
        # 4. Total Loss
        # ========================================
        total_loss = (
            self.lambda_roi * roi_loss +
            self.lambda_subject * subject_loss +
            self.lambda_consistency * consistency_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'roi_mse': roi_loss.item(),
            'subject_mse': subject_loss.item(),
            'consistency': consistency_loss.item()
        }
        
        return total_loss, loss_dict