# train_unet_roi.py 

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

from config_18roi import ROIBrainAgeConfig
from models.roi_model import ROIBrainAge
from json_18roi_dataset import MultimodalBrainAgeDataset, ROIVolumeDataset, collate_roi_batch  
from loss import ROIBrainAgeLoss 
from utils.training_utils import (
    set_seed, EMA, WarmupCosineSchedule, AverageMeter, save_checkpoint, load_checkpoint
    )
from utils.evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net ROI BrainAge")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def save_training_plots(history: dict, save_dir: Path):
    epochs = history['epoch']
    if len(epochs) < 2:
        return 

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })
    colors = {
        'train': '#2196F3',  # Blue
        'val':   '#F44336',  # Red
        'roi':   '#4CAF50',  # Green
        'subj':  '#FF9800',  # Orange
        'r':     '#9C27B0',  # Purple
    }

    # ============================================================
    # Plot 1: Loss Curves
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training / Validation Loss', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(epochs, history['train_loss'],
            color=colors['train'], linewidth=2, marker='o', markersize=3,
            label='Train Total Loss')
    if history['val_loss']:
        ax.plot(history['val_epochs'], history['val_loss'],
                color=colors['val'], linewidth=2, marker='s', markersize=4,
                label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history['train_roi_loss'],
            color=colors['roi'], linewidth=2, marker='o', markersize=3,
            label='Train ROI Loss')
    ax.plot(epochs, history['train_subject_loss'],
            color=colors['subj'], linewidth=2, marker='o', markersize=3,
            label='Train Subject Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Component Losses (Train)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_path = save_dir / 'loss_curves.png'
    plt.savefig(loss_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {loss_path}")

    # ============================================================
    # Plot 2: Metric Curves
    # ============================================================
    if not history['val_epochs']:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Validation Metrics', fontsize=14, fontweight='bold')

    val_ep = history['val_epochs']

    ax = axes[0]
    ax.plot(val_ep, history['val_roi_mae'],
            color=colors['roi'], linewidth=2, marker='s', markersize=4,
            label='Val ROI MAE')
    ax.plot(val_ep, history['val_subject_mae'],
            color=colors['val'], linewidth=2, marker='s', markersize=4,
            label='Val Subject MAE')

    best_mae = min(history['val_subject_mae'])
    best_ep  = val_ep[history['val_subject_mae'].index(best_mae)]
    ax.axhline(best_mae, color='gray', linestyle='--', alpha=0.7,
               label=f'Best Subject MAE: {best_mae:.3f}')
    ax.axvline(best_ep, color='gray', linestyle=':', alpha=0.5)
    ax.scatter([best_ep], [best_mae], color='red', zorder=5, s=80)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (years)')
    ax.set_title('Mean Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(val_ep, history['val_r'],
            color=colors['r'], linewidth=2, marker='s', markersize=4,
            label='Val Pearson r')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='r=0.8 target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pearson r')
    ax.set_title('Correlation (Pearson r)')
    ax.set_ylim([-1.0, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    metric_path = save_dir / 'metric_curves.png'
    plt.savefig(metric_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {metric_path}")


# ============================================================
# Train one epoch
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler,
                    loss_fn, scaler, ema, device, epoch, config):
    model.train()

    losses         = AverageMeter()
    roi_losses     = AverageMeter()
    subject_losses = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=config.fp16):
            outputs = model(batch)
            loss, loss_dict = loss_fn(outputs, batch)
            loss = loss / config.accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)

        losses.update(loss_dict['total'], n=batch['volume'].size(0))
        roi_losses.update(loss_dict.get('roi_mse', 0))         
        subject_losses.update(loss_dict.get('subject_mse', 0)) 

        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'roi':  f"{roi_losses.avg:.4f}",
            'subj': f"{subject_losses.avg:.4f}",
            'lr':   f"{scheduler.get_last_lr()[0]:.6f}"
        })

    return {
        'train_loss':         losses.avg,
        'train_roi_loss':     roi_losses.avg,
        'train_subject_loss': subject_losses.avg,
    }


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False

    # ── Config ──────────────────────────────────────────────
    config = UNetROIBrainAgeConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            for k, v in json.load(f).items():
                setattr(config, k, v)

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.save_dir, exist_ok=True)

    with open(Path(config.save_dir) / "config.json", 'w') as f:
        json.dump(
            {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
            f, indent=2, default=str
        )

    # ── Data ────────────────────────────────────────────────
    manifest_csv = config.manifest_csv
    split_json   = "/mnt/HDD16TB/Human_MRI/Multimodal_Human_BrainAge/Hybrid_UNETR_BrainAGE/subject_split.json"

    import nibabel as nib
    import torch.nn.functional as F

    label_map_img    = nib.load(config.label_map_path)
    label_map        = label_map_img.get_fdata().astype(np.int32)
    label_map_tensor = torch.from_numpy(label_map).unsqueeze(0).unsqueeze(0).float()
    label_map_resized = F.interpolate(
        label_map_tensor, size=config.target_shape, mode='nearest'
    ).squeeze(0).squeeze(0).numpy().astype(np.int32)

    print(f"Label map: {label_map.shape} → {label_map_resized.shape}")

    def make_base(split):
        return MultimodalBrainAgeDataset(
            manifest_csv=manifest_csv, split_json=split_json,
            split_name=split, target_shape=config.target_shape,
            cache_dir=config.cache_dir, use_cache=config.use_cache,
            max_ram_vols=config.max_ram_vols_per_worker
        )

    def make_roi_ds(base):
        return ROIVolumeDataset(
            base, label_map_resized,
            min_roi_voxels=config.min_roi_voxels,
            exclude_roi_ids=config.exclude_roi_ids,
            include_roi_ids=config.include_roi_ids,
            use_displacement=config.use_displacement,
        )

    train_ds = make_roi_ds(make_base('train'))
    val_ds   = make_roi_ds(make_base('val'))
    test_ds  = make_roi_ds(make_base('test'))

    roi_names = train_ds.get_roi_names()
    print(f"\nKey ROIs ({len(roi_names)}):")
    for roi_id, roi_name in sorted(roi_names.items()):
        voxels = np.sum(label_map_resized == roi_id)
        print(f"  [{roi_id:4d}] {roi_name:<35} ({voxels:,} voxels)")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=config.batch_size, shuffle=shuffle,
            num_workers=config.num_workers, collate_fn=collate_roi_batch,
            pin_memory=True, drop_last=shuffle,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    print(f"\nBatches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # ── Model ───────────────────────────────────────────────
    model     = UNetROIBrainAge(config).to(device)
    loss_fn   = ROIBrainAgeLoss(config)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    steps_per_epoch = len(train_loader) // config.accum_steps
    total_steps     = steps_per_epoch * config.epochs
    warmup_steps    = steps_per_epoch * config.warmup_epochs

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}")

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_scale=config.min_lr_scale
    )

    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    ema    = EMA(model, decay=config.ema_decay) if config.use_ema else None

    # ── Resume ──────────────────────────────────────────────
    start_epoch = 0
    best_metric = float('inf')

    if args.resume or config.auto_resume:
        resume_path = args.resume or (Path(config.save_dir) / "checkpoint_last.pt")
        if os.path.exists(resume_path):
            print(f"Resuming from: {resume_path}")
            ckpt = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, ema, device
            )
            start_epoch = ckpt['epoch'] + 1
            best_metric = ckpt.get('best_metric', float('inf'))
            print(f"  Epoch: {start_epoch}, Best MAE: {best_metric:.4f}")

    history = {
        'epoch':               [],
        'train_loss':          [],
        'train_roi_loss':      [],
        'train_subject_loss':  [],
        'val_epochs':          [],  
        'val_loss':            [],
        'val_roi_mae':         [],
        'val_subject_mae':     [],
        'val_r':               [],
    }

    # ── Training ────────────────────────────────────────────
    if not args.eval_only:
        patience = 10
        no_improve = 0

        for epoch in range(start_epoch, config.epochs):

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                loss_fn, scaler, ema, device, epoch, config
            )

            history['epoch'].append(epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_roi_loss'].append(train_metrics['train_roi_loss'])
            history['train_subject_loss'].append(train_metrics['train_subject_loss'])

            if (epoch + 1) % config.eval_every == 0:
                val_metrics, _ = evaluate_model(
                    model, val_loader, loss_fn, device,
                    use_amp=config.fp16, ema=ema,
                    use_ema=config.eval_use_ema,
                    desc=f"Val Epoch {epoch}"
                )

                history['val_epochs'].append(epoch)
                history['val_loss'].append(val_metrics.get('loss', 0.0))
                history['val_roi_mae'].append(val_metrics['roi_mae'])
                history['val_subject_mae'].append(val_metrics['subject_mae'])
                history['val_r'].append(val_metrics['global_r'])

                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss:      {train_metrics['train_loss']:.4f}")
                print(f"  Train ROI Loss:  {train_metrics['train_roi_loss']:.4f}")
                print(f"  Train Subj Loss: {train_metrics['train_subject_loss']:.4f}")
                print(f"  Val ROI MAE:     {val_metrics['roi_mae']:.4f}")
                print(f"  Val Subject MAE: {val_metrics['subject_mae']:.4f}")
                print(f"  Val r:           {val_metrics['global_r']:.4f}")

                if (val_metrics['subject_mae'] < best_metric
                            and val_metrics['global_r'] > 0.3):
                    best_metric = val_metrics['subject_mae']
                    no_improve = 0
                    save_checkpoint(
                        Path(config.save_dir) / "checkpoint_best.pt",
                        model, optimizer, scheduler, scaler, ema,
                        epoch, epoch * len(train_loader), best_metric, config
                    )
                    print(f"  ✓ Best! MAE={best_metric:.3f}  r={val_metrics['global_r']:.3f}")
                else:
                    no_improve += 1
                    print(f"  No improve ({no_improve}/{patience})", f"  r={val_metrics['global_r']:.3f}")
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                save_training_plots(history, Path(config.save_dir))

            if (epoch + 1) % config.save_every == 0:
                save_checkpoint(
                    Path(config.save_dir) / f"checkpoint_epoch_{epoch:03d}.pt",
                    model, optimizer, scheduler, scaler, ema,
                    epoch, epoch * len(train_loader), best_metric, config
                )

            save_checkpoint(
                Path(config.save_dir) / "checkpoint_last.pt",
                model, optimizer, scheduler, scaler, ema,
                epoch, epoch * len(train_loader), best_metric, config
            )

        save_training_plots(history, Path(config.save_dir))

        with open(Path(config.save_dir) / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  Saved: {Path(config.save_dir) / 'training_history.json'}")

    # ── Final Evaluation ────────────────────────────────────
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)

    best_ckpt = Path(config.save_dir) / "checkpoint_best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, ema=ema, device=device)

    test_metrics, predictions = evaluate_model(
        model, test_loader, loss_fn, device,
        use_amp=config.fp16, ema=ema,
        use_ema=config.eval_use_ema, desc="Test"
    )

    print("\nTest Results:")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    pred_dir = Path(config.save_dir) / "predictions"
    pred_dir.mkdir(exist_ok=True)

    with open(pred_dir / "test_metrics.json", 'w') as f:
        json.dump(
            {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
            f, indent=2
        )

    print(f"\nDone! Results saved to: {config.save_dir}")


if __name__ == "__main__":
    main()