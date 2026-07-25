# Drug-Target Interaction Predictor — Status Report
**Author:** Haseeb Ur Rehman
**Date:** July 25, 2026
**GitHub:** https://github.com/marine9056/drug-target-Predictor
**Target Rating:** 8.5/10 (per audit roadmap)

---

## 1. Current State Summary

| Area | Before (5.5/10) | Now | Notes |
|------|------------------|-----|-------|
| Architecture | 8/10 | 8.5/10 | 1D-CNN protein encoder, attention fusion tested |
| Prediction Accuracy | Unverified | ~7/10 | CI 0.765 (concat, prior run); current concat run undertrained |
| Code Quality | 7/10 | 8.5/10 | Fixed rdkit dep, hardened error handling, 28 tests |
| Documentation | 5/10 | 7/10 | README updated but needs final metric refresh |
| Testing | 6/10 | 8/10 | 28 tests passing, featurization + model coverage |
| Deployment | 2/10 | 6/10 | Dockerfile fixed, not yet deployed to HF Spaces |
| **Overall** | **5.5/10** | **~7.5/10** | Needs retrain + deployment to hit 8.5 |

---

## 2. Model Architecture

### Drug Encoder (GAT)
- **Type:** Graph Attention Network (3 layers, 4 heads)
- **Input:** 136-dim atom features from RDKit
- **Output:** 128-dim graph embedding via global mean pool
- **Parameters:** 85,120
- **Residual connections** + BatchNorm between layers

### Protein Encoder (1D-CNN)
- **Type:** Embedding + 2-layer 1D convolutional network
- **Input:** Integer-encoded amino acid sequence (max length 1000)
- **Architecture:** Embedding(64d) → Conv1d(5, 64) → BN → MaxPool → Conv1d(3, 128) → BN → AdaptiveMaxPool → FC(128→128)
- **Output:** 128-dim protein embedding
- **Parameters:** 63,680
- **Replaces** broken AA-composition encoder (was flattening 64K dims)

### Fusion Layer (Two variants tested)

| | Concat | Attention |
|---|---|---|
| Mechanism | Linear(256→256) | Multi-head cross-attention (drug=query, protein=key/value) |
| Parameters | 65,792 | 99,072 |
| Total model params | 247,617 | 280,897 |

### Prediction Head
- MLP: 256→128→128→1 (ReLU + Dropout 0.3)
- **Parameters:** 33,025

---

## 3. Dataset

- **Source:** Davis Kinase Dataset (MoleculeNet benchmark)
- **Size:** 29,444 drug-target pairs
- **Drugs:** 68 unique compounds
- **Proteins:** 443 unique kinase sequences
- **Target:** pKd binding affinity (log scale, range ~2–11)
- **Split:** 80% train (20,612) / 10% val (2,944) / 10% test (5,888)
- **Data file:** `data/raw/davis_full.csv`

---

## 4. Training Configuration

```yaml
batch_size: 256
epochs: 100
learning_rate: 0.001
weight_decay: 0.0001
patience: 10           # Early stopping
scheduler: plateau     # ReduceLROnPlateau
scheduler_factor: 0.5
scheduler_patience: 5
min_lr: 0.00001
optimizer: AdamW
loss: MSE
device: CPU (2 cores, no GPU)
num_workers: 0
```

---

## 5. Experiment Results

### Run 1: Concat Fusion (prior run, 17 epochs, early stopped at epoch 6)
| Metric | Value |
|--------|-------|
| MSE | 0.6065 |
| MAE | 0.4894 |
| CI (Concordance Index) | **0.7646** |
| Pearson | 0.5079 |
| Spearman | 0.4844 |
| Best Val Loss | 0.6029 |

### Run 2: Attention Fusion (9 epochs, early stopped at epoch 9)
| Metric | Value |
|--------|-------|
| MSE | 0.7168 |
| MAE | 0.6115 |
| CI (Concordance Index) | 0.6468 |
| Pearson | 0.2896 |
| Spearman | 0.2737 |
| Best Val Loss | 0.7440 |

### Run 3: Concat Fusion (latest run, 3 epochs, early stopped prematurely)
| Metric | Value |
|--------|-------|
| MSE | 0.8429 |
| MAE | 0.6159 |
| CI (Concordance Index) | 0.7481 |
| Pearson | 0.4810 |
| Spearman | 0.4552 |
| Best Val Loss | 0.8369 |

### Side-by-Side Comparison

| Metric | Concat (Run 1) | Attention (Run 2) | Concat (Run 3) | Winner |
|--------|----------------|-------------------|----------------|--------|
| MSE ↓ | **0.6065** | 0.7168 | 0.8429 | Concat (Run 1) |
| MAE ↓ | **0.4894** | 0.6115 | 0.6159 | Concat (Run 1) |
| CI ↑ | **0.7646** | 0.6468 | 0.7481 | Concat (Run 1) |
| Pearson ↑ | **0.5079** | 0.2896 | 0.4810 | Concat (Run 1) |
| Spearman ↑ | **0.4844** | 0.2737 | 0.4552 | Concat (Run 1) |
| Epochs trained | 17 | 9 | 3 | — |

### Published SOTA for reference
- **Best published CI on Davis:** ~0.88 (with pretrained protein embeddings + k-fold CV)
- **Our best CI:** 0.7646 (concat fusion, single split)

---

## 6. Key Findings

1. **Concat fusion outperforms attention fusion** on this dataset size (29K pairs, 68 drugs, 443 proteins). Attention needs more data or more epochs to learn meaningful cross-modal interactions.

2. **Concat fusion is more stable.** Run 1 achieved CI 0.765 with 17 epochs. Attention fusion achieved only CI 0.647 even after 9 epochs of full training.

3. **1D-CNN protein encoder is a major improvement** over the original AA-composition encoder (CI improved from ~0.71 to ~0.76 with concat).

4. **Run 3 (latest concat) is undertrained** — only 3 epochs before best model was saved. The model needs a full 15-20 epoch run to converge properly. Best model from Run 1 (CI 0.765) is the current benchmark.

5. **No GPU is the main bottleneck.** Training on CPU with 2 cores. Full 100-epoch run would take 6+ hours. Early stopping typically triggers at 15-20 epochs.

---

## 7. What's Done (8.5 Roadmap Checklist)

| # | Task | Status |
|---|------|--------|
| 1 | Fix `rdkit-pypi` → `rdkit` in requirements.txt and pyproject.toml | DONE |
| 2 | Dockerfile verified for HF Spaces | DONE |
| 3 | Test attention fusion as default | DONE (concat is better) |
| 4 | Retrain with new protein encoder | DONE (best: CI 0.765) |
| 5 | Compare fusion methods | DONE |
| 6 | Delete dead requirements-full.txt | DONE |
| 7 | Add tests for featurization + model | DONE (28 tests) |
| 8 | Harden data_loader.py error handling | DONE (raises RuntimeError) |
| 9 | Add GitHub Actions CI | DONE |
| 10 | Retrain concat to convergence | **TODO** |
| 11 | Update README with final metrics | **TODO** |
| 12 | Deploy on HuggingFace Spaces | **TODO** |
| 13 | Verify live demo works | **TODO** |

---

## 8. Remaining Questions for Improvement

1. **Should we retrain concat fusion to convergence?** Run 3 only reached epoch 3. A full run to 15-20 epochs should push CI from 0.748 to ~0.77-0.80.

2. **Should we try k-fold cross-validation?** Single 80/10/10 split may not be representative. 5-fold CV would give mean ± std for CI, which is more defensible.

3. **Should we add pretrained protein embeddings?** The audit report mentions ESM (protein language model) embeddings. The `use_esm` flag is stubbed in featurization.py but not implemented. This could push CI closer to 0.85+.

4. **Should we add prediction confidence (MC Dropout)?** The audit recommends uncertainty estimation. A bare pKd number without confidence bands is misleading for a scientific tool.

5. **Should we add data augmentation or regularization?** The model may benefit from mixup, label smoothing, or dropout tuning to improve generalization.

6. **Learning rate schedule:** Current ReduceLROnPlateau with factor=0.5 and patience=5 may be too aggressive. Cosine annealing might give smoother convergence.

7. **Protein encoder depth:** 2-layer CNN may underfit. A deeper encoder (4 layers) with residual connections could capture more complex motifs.

8. **Drug encoder:** GAT with 3 layers may be overkill for molecules with ~50 atoms. A 2-layer GAT or adding virtual nodes could help.

---

## 9. Files and Checkpoints

```
models/checkpoints/
├── best_model.pt              # Latest best (concat, undertrained - epoch 3)
├── best_model_concat.pt       # Concat fusion checkpoint (epoch 3, val_loss=0.837)
└── best_model_attention.pt    # Attention fusion checkpoint (epoch 9, val_loss=0.744)
```

**Best model to deploy:** The Run 1 concat model (CI 0.765) was overwritten. We need one final retrain to produce a properly converged concat checkpoint.

---

## 10. Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.1, PyTorch Geometric 2.4 |
| Chemistry | RDKit (SMILES → molecular graphs) |
| Protein Encoding | Custom 1D-CNN (embedding + conv layers) |
| Drug Encoding | GAT (Graph Attention Network) |
| Web Framework | Streamlit + Plotly |
| Containerization | Docker (Python 3.10-slim) |
| CI/CD | GitHub Actions (pytest on push) |
| Deployment Target | HuggingFace Spaces |
| Dataset | Davis Kinase (MoleculeNet) |
| License | MIT |
