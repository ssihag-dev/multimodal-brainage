# multimodal-brainage
Region-wise BrainAGE prediction using ROI pooling, 3D CNN feature extraction, and Transformer-based aggregation for interpretable brain age estimation from MRI.
# Region-wise BrainAGE Prediction

## Overview

This project implements a neuroanatomically grounded framework for **Brain Age Prediction (BrainAGE)** using **T1-weighted MRI scans**.

Unlike conventional BrainAGE models that predict a single global brain age, this framework generates both:

- Subject-level brain age predictions
- ROI-level brain age predictions across 18 anatomically defined brain regions

The model combines:

- Hierarchical 3D CNN feature extraction
- ROI pooling
- Transformer-based ROI aggregation
- Multi-task learning for regional and global age prediction

---

## Dataset

The model was developed using a multi-cohort dataset containing **2,851 T1-weighted MRI scans** from:

- ADNI
- AIBL
- OASIS
- IXI
- SALD
- DLBS
- CoRR
- SchizConnect
- NIFD
- PPMI
- BGSP

---

## Model Architecture

The proposed framework consists of:

1. **Hierarchical 3D CNN Encoder**
2. **ROI Pooling on 18 literature-grounded brain regions**
3. **ROI Feature Encoder**
4. **Transformer Aggregation Network**
5. **ROI-Level and Subject-Level Age Prediction Heads**

---

## Key ROIs

The model focuses on anatomically defined brain regions including:

- Hippocampus
- Amygdala
- Lateral Ventricle
- Thalamus
- Caudate
- Putamen
- Superior Frontal Cortex
- Middle Temporal Cortex
- Insula

---

## Training

Run training using:

```bash
python train_18roi.py
```

Configuration settings can be modified in:

```bash
config_18roi.py
```

---

## Results

| Metric | Value |
|--------|-------|
| Subject-level MAE | 8.24 years |
| Subject-level R² | 0.89 |
| ROI-level MAE | 9.37 years |
| ROI-level R² | 0.86 |

---

## Authors

- Seoyoon Jeong
- Celio Boulay
- Sakshi Sihag

---

## Citation

**BMEN4460 Final Project, Columbia University**

*"Region-wise BrainAGE Prediction"*
