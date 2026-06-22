from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scipy.stats import pearsonr


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    """
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r, _ = pearsonr(predictions, targets)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r': float(r)
    }


@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device,
                   use_amp=True, ema=None, use_ema=False, desc="Eval"):
    model.eval()
    if use_ema and ema is not None:
        ema.apply_shadow(model)

    all_subject_preds   = []
    all_subject_targets = []
    all_roi_preds       = []
    all_roi_targets     = []
    all_attention       = []
    all_roi_id_lists    = []
    losses = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch)
            if loss_fn is not None:
                loss, _ = loss_fn(outputs, batch)
                losses.append(loss.item())

        subject_preds   = outputs['subject_ages'].cpu().numpy()
        subject_targets = batch['age'].cpu().numpy()

        all_subject_preds.extend(subject_preds.tolist())
        all_subject_targets.extend(subject_targets.tolist())

        # ROI-level
        for b_idx, roi_ages in enumerate(outputs['roi_ages']):
            roi_preds_np = roi_ages.cpu().numpy()
            true_age     = float(batch['age'][b_idx].cpu())
            all_roi_preds.extend(roi_preds_np.tolist())
            all_roi_targets.extend([true_age] * len(roi_preds_np))

        for attn in outputs['attention_weights']:
            all_attention.append(attn.cpu().numpy())

        all_roi_id_lists.extend(outputs['roi_id_lists'])

    if use_ema and ema is not None:
        ema.restore(model)

    sp = np.array(all_subject_preds)
    st = np.array(all_subject_targets)
    rp = np.array(all_roi_preds)
    rt = np.array(all_roi_targets)

    def safe_r(x, y):
        if np.std(x) < 1e-6:
            return float('nan')
        r, _ = pearsonr(x, y)
        return float(r)

    metrics = {
        'loss':         np.mean(losses) if losses else 0.0,
        'subject_mae':  float(np.mean(np.abs(sp - st))),
        'subject_rmse': float(np.sqrt(np.mean((sp - st)**2))),
        'subject_r':    safe_r(sp, st),
        'subject_bias': float(np.mean(sp - st)),
        'roi_mae':      float(np.mean(np.abs(rp - rt))),
        'roi_r':        safe_r(rp, rt),
        'mae':          float(np.mean(np.abs(sp - st))),
        'global_r':     safe_r(sp, st),
    }

    predictions = {
        'subject_preds':   sp,
        'subject_targets': st,
        'roi_preds':       rp,
        'roi_targets':     rt,
        'attention':       all_attention,
        'roi_id_lists':    all_roi_id_lists,
    }

    return metrics, predictions


def aggregate_roi_to_subject(
    roi_level_data: List[Dict[str, Any]],
    methods: List[str] = None
) -> Dict[str, Dict[str, float]]:
    
    if methods is None:
        methods = ['mean', 'median', 'trimmed_mean']
    
    from collections import defaultdict
    
    # Group by sample_id
    grouped = defaultdict(list)
    for row in roi_level_data:
        grouped[row['sample_id']].append(row)
    
    results_by_method = {}
    
    for method in methods:
        subject_preds = []
        subject_targets = []
        
        for sample_id, rows in grouped.items():
            # Aggregate ROI predictions for this subject
            roi_preds = [row['pred_age'] for row in rows]
            true_age = rows[0]['age']
            
            if method == 'mean':
                subj_pred = np.mean(roi_preds)
            elif method == 'median':
                subj_pred = np.median(roi_preds)
            elif method == 'trimmed_mean':
                trim_frac = 0.2
                if len(roi_preds) < 3:
                    subj_pred = np.mean(roi_preds)
                else:
                    trim_count = int(np.floor(len(roi_preds) * trim_frac))
                    sorted_preds = sorted(roi_preds)
                    if trim_count > 0:
                        sorted_preds = sorted_preds[trim_count:-trim_count]
                    subj_pred = np.mean(sorted_preds)
            
            subject_preds.append(subj_pred)
            subject_targets.append(true_age)
        
        # Compute metrics
        subject_preds = np.array(subject_preds)
        subject_targets = np.array(subject_targets)
                
        mae = float(np.mean(np.abs(subject_preds - subject_targets)))
        rmse = float(np.sqrt(np.mean((subject_preds - subject_targets) ** 2)))
        r, _ = pearsonr(subject_preds, subject_targets)
        bias = float(np.mean(subject_preds - subject_targets))
        
        results_by_method[method] = {
            'mae': mae,
            'rmse': rmse,
            'pearson_r': float(r),
            'bias': bias
        }
    
    return results_by_method