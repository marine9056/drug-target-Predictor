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

| File | Provenance | Epoch | CI | Pearson | MSE | Status |
|------|-----------|-------|-----|---------|-----|--------|
| best_model.pt | CPU, concat | 16 | 0.772 | 0.540 | 0.693 | **Active** (best verified) |
| best_model_concat.pt | CPU, concat | 16 | 0.772 | 0.540 | 0.693 | Same as best_model.pt |
| best_model_attention.pt | CPU, attention | 9 | 0.647 | 0.290 | 0.717 | Attention variant |
| GPU (Colab, downloaded) | GPU, concat | 8 | 0.756 | 0.500 | 0.777 | Worse — early stopped at epoch 8 |

**Key findings:**
- GPU training (Colab T4, 100 epochs) stopped at epoch 8 (early stopping, patience=10)
- GPU model is **worse** than CPU model on all metrics — the 100-epoch training didn't converge
- **PyG version mismatch:** Colab uses newer PyG with `lin_src`/`lin_dst` keys; local uses `lin` — requires key remapping for checkpoint loading
- Previous "CI 0.846" claim was not reproducible — likely training metrics, not test metrics

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
| **1D-CNN full convergence** | **Concat** | **CPU** | **16** | **0.772** | **0.540** | **0.497** | **0.693** |
| Attention fusion | Attention | CPU | 9 | 0.647 | 0.290 | 0.274 | 0.717 |
| GPU (early stopped) | Concat | T4 | 8 | 0.756 | 0.500 | 0.468 | 0.777 |

### Published SOTA for reference
- Best published CI on Davis: ~0.88 (pretrained protein embeddings + k-fold CV)
- Our verified CI: 0.772 (CPU, single split)
- Our reported CI: 0.846 (GPU, pending local verification)

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
| 10 | Retrain concat to convergence | DONE (CI 0.772, epoch 6) |
| 11 | Update README with honest metrics + Known Limitations | DONE |
| 12 | Fix app_deploy.py fake predictions | DONE |
| 13 | Retrain on GPU (download checkpoint) | **TODO** |
| 14 | Deploy on HuggingFace Spaces | **TODO** |
| 15 | Verify live demo works | **TODO** |

---

## 6. Known Limitations

- **MSE vs ranking tradeoff:** Model compresses predictions (4.3-8.0) while labels span 5.0-10.4
- **High-pKd bias:** Under-predicts high-affinity pairs (pKd > 9.0)
- **Single dataset:** Davis Kinase only, no cross-family validation
- **No adversarial robustness testing**
- **CPU-only verified:** GPU checkpoint needs re-downloading

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
