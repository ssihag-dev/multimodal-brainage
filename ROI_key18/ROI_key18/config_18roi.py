# config_unet_roi.py

from dataclasses import dataclass, field
from typing import Tuple, Optional
import os

# ============================================================
# FreeSurfer aparc+aseg+hsf Key ROIs for BrainAge
# Reference: Cole et al. 2017, Franke & Gaser 2019,
#            Fjell et al. 2009, Dickerson et al. 2001
# ============================================================
BRAINAGE_KEY_ROI_IDS = (
    # ── Hippocampus ─────────────────────────────────────────
    # Jack et al. (2000) Neurology: 1.73% annual atrophy (healthy elderly)
    # Raz et al. (2005) Cereb Cortex: core biomarker of aging
    17, 53,

    # ── Amygdala ────────────────────────────────────────────
    # Peng et al. (2023) eLife: key BrainAge region based on SHAP
    # Brabec et al. (2010) Neurobiol Aging: volume reduction with aging
    18, 54,

    # ── Lateral Ventricle ───────────────────────────────────
    # Scahill et al. (2003) Cereb Cortex: 0.65 cm³ annual expansion
    # Resnick et al. (2000) Cereb Cortex: indirect marker of aging
    4, 43,

    # ── Thalamus ────────────────────────────────────────────
    # Walhovd et al. (2011) Neurobiol Aging: aging of subcortical structures
    # Fjell et al. (2013) NeuroImage: volume reduction with aging
    10, 49,

    # ── Striatum ────────────────────────────────────────────
    # Fjell et al. (2013) NeuroImage: dopamine system decline with aging
    # Raz et al. (2003) Cereb Cortex: aging effects in caudate/putamen
    11, 50,   # Caudate
    12, 51,   # Putamen

    # ── Cortical: Frontal ───────────────────────────────────
    # Fjell et al. (2009) Cereb Cortex: frontal cortex shows the
    #   strongest aging effects (superior > inferior frontal)
    1028, 2028,   # superior-frontal

    # ── Cortical: Temporal ──────────────────────────────────
    # Fjell et al. (2009) Cereb Cortex: cortical thinning in temporal regions
    # Peng et al. (2023) eLife: superior temporal as a key indicator
    1015, 2015,   # middle-temporal

    # ── Insula ──────────────────────────────────────────────
    # Peng et al. (2023) eLife: key BrainAge region shared across sexes
    # Fjell et al. (2014) J Neurosci: aging-related changes observed
    1035, 2035,
)


@dataclass
class ROIBrainAgeConfig:

    # ==================== Paths ====================
    subject_csv: str = "/mnt/HDD16TB/Human_MRI/Multimodal_Human_BrainAge/dataset/XYF_aging_age_sex_info.xlsx"
    manifest_csv: str = "/mnt/HDD16TB/Human_MRI/Multimodal_Human_BrainAge/Hybrid_UNETR_BrainAGE/paired_subject_manifest.csv"
    t1_dir: str = "/mnt/HDD16TB/Human_MRI/Multimodal_Human_BrainAge/dataset/Step7_ANTs/Warped"
    disp_dir: str = "/mnt/HDD16TB/Human_MRI/Multimodal_Human_BrainAge/dataset/Step7_ANTs/Displacement"
    label_map_path: str = "/mnt/HDD16TB/Human_MRI/Multimodal_Human_BrainAge/dataset/aparc+aseg+hsf_XYF_Aging_18_to_23_T1w_template_nn_labelmap.nii.gz"

    save_dir: str = "/mnt/HDD16TB/Human_MRI/Multimodal_Human_BrainAge/Hybrid_UNETR_BrainAGE/output_unet_18roi"
    cache_dir: str = "/mnt/HDD16TB/Human_MRI/Multimodal_Human_BrainAge/Hybrid_UNETR_BrainAGE/output_unet_roi/cache"

    # ==================== Input ====================
    target_shape: Tuple[int, int, int] = (160, 192, 160)
    in_channels: int = 1                    
    use_displacement: bool = False          

    # ==================== U-Net Architecture ====================
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    use_batch_norm: bool = True
    dropout: float = 0.1

    roi_pool_level: int = 3
    roi_pool_feature_dim: int = 256        

    # ==================== ROI ====================
    roi_pool_size: Tuple[int, int, int] = (4, 4, 4)
    min_roi_voxels: int = 64
    exclude_roi_ids: Tuple[int, ...] = (0,)
    include_roi_ids: Tuple[int, ...] = BRAINAGE_KEY_ROI_IDS  

    # ==================== ROI Feature Encoder ====================
    roi_feature_dim: int = 512

    # ==================== Transformer ====================
    transformer_dim: int = 512
    transformer_depth: int = 2
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.1

    # ==================== Loss ====================
    lambda_roi: float = 1.0
    lambda_subject: float = 1.0
    lambda_consistency: float = 0.0  #None for this time

    # ==================== Training ====================
    seed: int = 42
    epochs: int = 30
    batch_size: int = 4
    accum_steps: int = 4                  
    num_workers: int = 8

    # Optimization
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # LR Schedule
    warmup_epochs: int = 5
    min_lr_scale: float = 0.1

    # Mixed precision
    fp16: bool = True

    # ==================== EMA ====================
    use_ema: bool = True
    ema_decay: float = 0.999
    eval_use_ema: bool = True

    # ==================== Data Splits ====================
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ==================== Caching ====================
    use_cache: bool = True
    max_ram_vols_per_worker: int = 5

    # ==================== Checkpointing ====================
    eval_every: int = 1
    save_every: int = 3
    auto_resume: bool = True

    def __post_init__(self):
        assert self.roi_pool_feature_dim == self.encoder_channels[self.roi_pool_level], \
            f"roi_pool_feature_dim {self.roi_pool_feature_dim} != " \
            f"encoder_channels[{self.roi_pool_level}] {self.encoder_channels[self.roi_pool_level]}"

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)