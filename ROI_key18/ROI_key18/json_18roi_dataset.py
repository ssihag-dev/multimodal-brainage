# data/json_roi_dataset.py

import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.ndimage import zoom


ROI_NAME_MAP = {
    2:    "Left-Cerebral-WM",
    4:    "Left-Lateral-Ventricle",
    10:   "Left-Thalamus",
    11:   "Left-Caudate",
    12:   "Left-Putamen",
    13:   "Left-Pallidum",
    17:   "Left-Hippocampus",
    18:   "Left-Amygdala",
    41:   "Right-Cerebral-WM",
    43:   "Right-Lateral-Ventricle",
    49:   "Right-Thalamus",
    50:   "Right-Caudate",
    51:   "Right-Putamen",
    52:   "Right-Pallidum",
    53:   "Right-Hippocampus",
    54:   "Right-Amygdala",
    1003: "Left-caudal-middle-frontal",
    1007: "Left-fusiform",
    1015: "Left-middle-temporal",
    1028: "Left-superior-frontal",
    1035: "Left-insula",
    2003: "Right-caudal-middle-frontal",
    2007: "Right-fusiform",
    2015: "Right-middle-temporal",
    2028: "Right-superior-frontal",
    2035: "Right-insula",
}


class MultimodalBrainAgeDataset(Dataset):
    
    def __init__(
        self,
        manifest_csv: str,
        split_json: str,
        split_name: str,  # 'train', 'val', or 'test'
        target_shape: Tuple[int, int, int] = (160, 192, 160),
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        max_ram_vols: int = 2
    ):
        
        super().__init__()
        
        self.target_shape = target_shape
        self.split_name = split_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and cache_dir is not None
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_df = pd.read_csv(manifest_csv)
        with open(split_json, 'r') as f:
            split_data = json.load(f)
        
        sample_ids = split_data['samples'][split_name]
        self.records = manifest_df[
            manifest_df['sample_id'].isin(sample_ids)
        ].to_dict('records')
        
        self.max_ram_vols = max_ram_vols
        self.ram_cache = {}
        self.cache_order = []
        
        print(f"\nMultimodalBrainAgeDataset [{split_name}]:")
        print(f"  Samples: {len(self.records)}")
        print(f"  Target shape: {target_shape}")
        print(f"  Cache: {use_cache}")
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample_id = record['sample_id']
        
        if sample_id in self.ram_cache:
            return self.ram_cache[sample_id]
        
        if self.use_cache:
            cache_path = self.cache_dir / f"{sample_id}.pt"
            
            if cache_path.exists():
                try:
                    cached_data = torch.load(cache_path, map_location='cpu')
                    self._update_ram_cache(sample_id, cached_data)
                    return cached_data
                except Exception as e:
                    print(f"Warning: Failed to load cache for {sample_id}: {e}")
        
        volume = self._load_volume(record)
        
        data = {
            'volume': volume,
            'age': float(record['age']),
            'subject_id': str(record['subject_id']),
            'sample_id': sample_id
        }
        
        if self.use_cache:
            cache_path = self.cache_dir / f"{sample_id}.pt"
            try:
                import fcntl
                lock_path = cache_path.with_suffix('.lock')
                
                with open(lock_path, 'w') as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    
                    if not cache_path.exists():
                        torch.save(data, cache_path)
                    
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                
                lock_path.unlink(missing_ok=True)
                
            except Exception as e:
                print(f"Warning: Failed to cache {sample_id}: {e}")
        
        self._update_ram_cache(sample_id, data)
        
        return data
    
    def _load_volume(self, record: Dict[str, Any]) -> torch.Tensor:
        # ==================== Load T1 ====================
        t1_img = nib.load(record['t1_path'])
        t1_data = t1_img.get_fdata(dtype=np.float32)
        
        if t1_data.ndim == 4:
            t1_data = t1_data[..., 0]
        
        if t1_data.ndim != 3:
            raise ValueError(f"T1 must be 3D, got {t1_data.ndim}D: {t1_data.shape}")
        
        # ==================== Load Displacement ====================
        disp_img = nib.load(record['inversewarp_path'])
        disp_data = disp_img.get_fdata(dtype=np.float32)
        
        if disp_data.ndim == 5:
            disp_data = disp_data[..., 0, :]
        elif disp_data.ndim == 4:
            pass
        else:
            raise ValueError(
                f"Unexpected displacement dimensions: {disp_data.ndim}D, shape {disp_data.shape}"
            )
        
        if disp_data.shape[-1] != 3:
            raise ValueError(
                f"Displacement should have 3 channels (dx,dy,dz), got {disp_data.shape[-1]}"
            )
        
        # ==================== Resize T1 ====================
        if t1_data.shape != self.target_shape:
            t1_data = self._resize_volume(t1_data, self.target_shape)
        
        # ==================== Resize Displacement ====================
        if disp_data.shape[:3] != self.target_shape:          
            zoom_factors = [
                self.target_shape[i] / disp_data.shape[i]
                for i in range(3)
            ]
            zoom_factors.append(1.0)  
            disp_data = zoom(disp_data, zoom_factors, order=1)
        
        assert t1_data.shape == self.target_shape, \
            f"T1 shape mismatch: {t1_data.shape} != {self.target_shape}"
        
        assert disp_data.shape == (*self.target_shape, 3), \
            f"Displacement shape mismatch: {disp_data.shape} != {(*self.target_shape, 3)}"
        
        # ==================== Normalize T1 ====================
        # 99th percentile normalization
        mask = t1_data > 0
        if mask.any():
            p99 = np.percentile(t1_data[mask], 99)
            t1_data = np.clip(t1_data / (p99 + 1e-8), 0, 1)
        else:
            t1_data = np.zeros_like(t1_data)
        
        # ==================== Normalize Displacement ====================
        disp_normalized = np.zeros_like(disp_data)
        
        for i in range(3):
            channel = disp_data[..., i]
            
            # Z-score normalization per channel
            mean = channel.mean()
            std = channel.std()
            
            if std > 1e-8:
                disp_normalized[..., i] = (channel - mean) / std
            else:
                disp_normalized[..., i] = 0.0
        
        # ==================== Concatenate ====================
        # Combine T1 and displacement: [D, H, W, 4]
        volume_np = np.concatenate([
            t1_data[..., np.newaxis], 
            disp_normalized            
        ], axis=-1)
        
        volume = torch.from_numpy(volume_np).permute(3, 0, 1, 2).float()
        
        assert volume.shape == (4, *self.target_shape), \
            f"Final volume shape mismatch: {volume.shape} != {(4, *self.target_shape)}"
        
        return volume
    
    
    def _resize_volume(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:        
        if volume.ndim != 3:
            raise ValueError(
                f"_resize_volume expects 3D input, got {volume.ndim}D with shape {volume.shape}"
            )
        
        zoom_factors = [
            target_shape[i] / volume.shape[i]
            for i in range(3)
        ]
        
        resized = zoom(volume, zoom_factors, order=1)
        assert resized.shape == target_shape, \
            f"Resize failed: output shape {resized.shape} != target {target_shape}"
        
        return resized
    
    def _update_ram_cache(self, sample_id: str, data: Dict[str, Any]):
        if sample_id in self.ram_cache:
            return
        
        if len(self.ram_cache) >= self.max_ram_vols:
            oldest_id = self.cache_order.pop(0)
            del self.ram_cache[oldest_id]
        
        self.ram_cache[sample_id] = data
        self.cache_order.append(sample_id)


class ROIVolumeDataset(Dataset):
    
    def __init__(
        self,
        base_dataset: Dataset,
        label_map: np.ndarray,         
        min_roi_voxels: int = 64,
        exclude_roi_ids: Optional[List[int]] = None,
        include_roi_ids: Optional[Tuple[int, ...]] = None, 
        use_displacement: bool = False,                    
        augment=False,   
    ):
        self.base_dataset = base_dataset
        self.label_map = label_map
        self.min_roi_voxels = min_roi_voxels
        self.exclude_roi_ids = set(exclude_roi_ids) if exclude_roi_ids else {0}
        self.include_roi_ids = set(include_roi_ids) if include_roi_ids else None
        self.use_displacement = use_displacement
        self.augment = augment
        self.valid_roi_ids = self._discover_valid_rois()

        print(f"[ROIVolumeDataset] Input: {'T1 + Displacement' if use_displacement else 'T1 only'}")
        print(f"[ROIVolumeDataset] Valid ROIs: {len(self.valid_roi_ids)}")
        if self.include_roi_ids:
            missing = self.include_roi_ids - set(self.valid_roi_ids)
            if missing:
                print(f"[ROIVolumeDataset] Warning: {len(missing)} requested ROIs not found in label map: {missing}")

    def _discover_valid_rois(self) -> List[int]:
        unique_ids = np.unique(self.label_map)
        valid_ids = []

        for roi_id in unique_ids:
            if roi_id in self.exclude_roi_ids:
                continue

            if self.include_roi_ids is not None:
                if roi_id not in self.include_roi_ids:
                    continue

            if np.sum(self.label_map == roi_id) >= self.min_roi_voxels:
                valid_ids.append(int(roi_id))

        return sorted(valid_ids)

    def get_roi_names(self) -> Dict[int, str]:
        return {
            roi_id: ROI_NAME_MAP.get(roi_id, f"ROI_{roi_id}")
            for roi_id in self.valid_roi_ids
        }
        
    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            volume = torch.flip(volume, dims=[3])   

        noise = torch.randn_like(volume) * 0.02
        volume = volume + noise
        volume = torch.clamp(volume, 0.0, 1.0)

        return volume

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        volume = sample['volume']  

        if self.use_displacement:
            pass  
        else:
            volume = volume[:1] 

        if self.augment:
            volume = self._augment(volume)
        label_map = torch.from_numpy(self.label_map).long()
        return {
                'volume': volume,
                'label_map': label_map,
                'roi_list': self.valid_roi_ids,
                'age': sample['age'],
                'subject_id': sample['subject_id'],
            }


def collate_roi_batch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for ROI volume batches."""
    volumes = torch.stack([item['volume'] for item in batch_list], dim=0)
    label_maps = torch.stack([item['label_map'] for item in batch_list], dim=0)
    roi_lists = [item['roi_list'] for item in batch_list]
    ages = torch.tensor([item['age'] for item in batch_list], dtype=torch.float32)
    subject_ids = [item['subject_id'] for item in batch_list]

    return {
        'volume': volumes,           # [B, 1 or 4, D, H, W]
        'label_map': label_maps,     # [B, D, H, W]
        'roi_lists': roi_lists,      # List[List[int]], length B
        'age': ages,                 # [B]
        'subject_ids': subject_ids,  # List[str], length B
    }