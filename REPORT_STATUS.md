# Drug-Target Interaction Predictor — Status Report
**Author:** Haseeb Ur Rehman
**Date:** July 25, 2026
**GitHub:** https://github.com/marine9056/drug-target-Predictor
**Target Rating:** 8.5/10 (per audit roadmap)

---

## 1. Current State Summary

| Area | Before (5.5/10) | Now | Notes |
|------|------------------|-----|-------|
| Architecture | 8/10 | 8.5/10 | 1D-CNN protein encoder, concat fusion |
| Prediction Accuracy | Unverified | 7.5/10 | Verified CI 0.772 (CPU); GPU 0.846 pending re-verification |
| Code Quality | 7/10 | 8.5/10 | Fixed rdkit dep, hardened error handling, 28 tests |
| Documentation | 5/10 | 8/10 | README honest, Known Limitations added |
| Testing | 6/10 | 8/10 | 28 tests passing |
| Deployment | 2/10 | 6/10 | Dockerfile fixed, not yet deployed to HF Spaces |
| **Overall** | **5.5/10** | **~7.5/10** | Needs GPU retrain + HF deployment |

---

## 2. Verified Checkpoint Status

| File | Provenance | Epoch | CI | Pearson | MSE (calib) | Status |
|------|-----------|-------|-----|---------|-------------|--------|
| best_model.pt | GPU (Colab T4, seed=42) | 33 | 0.793 | 0.606 | ~0.51 | **Active** (best verified) |
| best_model_cpu_ep16.pt | CPU, concat | 16 | 0.772 | 0.540 | 0.693 | CPU backup |
| best_model_concat.pt | CPU, concat | 16 | 0.772 | 0.540 | 0.693 | Same as CPU backup |
| best_model_attention.pt | CPU, attention | 9 | 0.647 | 0.290 | 0.717 | Attention variant |

**Key findings:**
- GPU training (Colab T4, 100 epochs, seed=42) — best at epoch 33
- GPU model improves ranking metrics: CI 0.772→0.793, Pearson 0.540→0.606, Spearman 0.497→0.534
- **Post-hoc calibration added** (models/checkpoints/calibration.json, slope 0.977, intercept 0.834, fitted on training split only): corrects systematic under-prediction, RMSE 1.03 → 0.71 (test), ranking metrics unchanged. Raw uncalibrated MSE was 1.035.
- **PyG version compatibility handled** in predict.py and streamlit_app.py (lin vs lin_src/lin_dst)

---

## 3. Training Configuration

```yaml
batch_size: 256
epochs: 100
learning_rate: 0.001
weight_decay: 0.0001
patience: 10
scheduler: plateau
scheduler_factor: 0.5
scheduler_patience: 5
min_lr: 0.00001
optimizer: AdamW
loss: MSE
device: CPU (2 cores) / GPU (T4 for Colab runs)
num_workers: 0
fusion: concat
```

---

## 4. Experiment Results

| Experiment | Fusion | HW | Epochs | CI | Pearson | Spearman | MSE |
|-----------|--------|-----|--------|-----|---------|----------|-----|
| Baseline (AA-comp) | Concat | CPU | 17 | 0.709 | 0.411 | 0.386 | 0.931 |
| 1D-CNN protein enc | Concat | CPU | 17 | 0.765 | 0.508 | 0.484 | 0.607 |
| 1D-CNN full convergence | Concat | CPU | 16 | 0.772 | 0.540 | 0.497 | 0.693 |
| Attention fusion | Attention | CPU | 9 | 0.647 | 0.290 | 0.274 | 0.717 |
| **GPU convergence (best)** | **Concat** | **T4** | **33** | **0.793** | **0.606** | **0.534** | **~0.51 (calib)** |

### Published SOTA for reference
- Best published CI on Davis: ~0.88 (pretrained protein embeddings + k-fold CV)
- Our verified CI: 0.793 (GPU, single split, seed=42)
- Raw (uncalibrated) MSE was 1.035; calibration reduces test RMSE to ~0.71

---

## 5. 8.5 Roadmap Checklist

| # | Task | Status |
|---|------|--------|
| 1 | Fix rdkit-pypi → rdkit | DONE |
| 2 | Dockerfile verified for HF Spaces | DONE |
| 3 | Test attention fusion as default | DONE (concat better) |
| 4 | Retrain with 1D-CNN protein encoder | DONE |
| 5 | Compare fusion methods | DONE |
| 6 | Delete dead requirements-full.txt | DONE |
| 7 | Add tests for featurization + model | DONE (28 tests) |
| 8 | Harden data_loader.py error handling | DONE |
| 9 | Add GitHub Actions CI | DONE |
| 10 | Retrain concat to convergence on GPU | DONE (CI 0.793, epoch 33) |
| 11 | Update README with honest metrics + Known Limitations | DONE |
| 12 | Fix app_deploy.py fake predictions | DONE |
| 13 | Add seed + higher patience for reproducible training | DONE |
| 14 | PyG version compatibility in predict.py/app | DONE |
| 15 | Deploy on Streamlit Community Cloud | DONE (live app deployed) |
| 16 | Verify live demo works | **TODO** (verify after this push auto-redeploys) |
| 17 | Fix binding classification (was inverted: low pKd shown as "Strong") | DONE |
| 18 | Add protein input validation (garbage/empty sequences) | DONE |
| 19 | Add post-hoc prediction calibration | DONE (RMSE 1.03→0.71) |

---

## 6. Known Limitations

- **Absolute accuracy:** Calibrated RMSE ~0.71 pKd units; individual predictions can be off by 1-2 units. Ranking (CI 0.793) is more reliable than absolute values.
- **High-pKd bias:** Under-predicts the strongest binders (pKd > 9.0) even after calibration — Davis has few very-high-affinity pairs.
- **Single dataset:** Davis Kinase only, no cross-family validation
- **No adversarial robustness testing**

---

## 7. Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.1, PyTorch Geometric 2.4 |
| Chemistry | RDKit |
| Protein Encoding | 1D-CNN (embedding + conv layers) |
| Drug Encoding | GAT (Graph Attention Network) |
| Web Framework | Streamlit + Plotly |
| Containerization | Docker (Python 3.10-slim) |
| CI/CD | GitHub Actions (pytest on push) |
| Deployment Target | HuggingFace Spaces |
| Dataset | Davis Kinase (MoleculeNet) |
| License | MIT |
