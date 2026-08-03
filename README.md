# Drug-Target Interaction Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/pyg-2.4+-ff6f00.svg)](https://pyg.org/)

A Graph Neural Network model that predicts binding affinity between drug molecules and protein targets, trained on the **Davis Kinase Dataset** (29,444 drug-target pairs).

## Live Demo

🔗 **[Try it live](https://drug-target-predictor-7g6gpzipeckyrwxlabyxwk.streamlit.app/)**

## Overview

Drug-target interaction prediction is a core problem in computational drug discovery. This project uses:

- **Graph Attention Networks (GAT)** to encode drug molecules from SMILES strings as molecular graphs
- **1D-CNN protein encoder** to capture sequence order and local motifs from amino acid sequences
- **Concatenation-based fusion** to combine drug and protein embeddings
- **MLP prediction head** to output binding affinity (pKd values)

### Model Architecture

```
Drug (SMILES) -> Molecular Graph -> GAT Layers (3x) -> Drug Embedding (128-d)
                                                                |
                                                           Concatenate
                                                                |
Protein (Sequence) -> Integer Encoding -> 1D-CNN -> Protein Embedding (128-d)
                                                                |
                                                           MLP Head -> pKd
```

## Results

### Verified Results (GPU-trained, local verification)

Trained on Davis Kinase Dataset (29,444 pairs, 80/10/10 split, seed=42).
Model trained for 100 epochs on NVIDIA T4 (Google Colab), best at epoch 33.
Verified locally on the same test split (calibrated):

| Metric | Value |
|--------|-------|
| Concordance Index (CI) | **0.793** |
| Pearson Correlation | **0.606** |
| Spearman Correlation | **0.534** |
| MSE (calibrated) | ~0.51 |
| RMSE (calibrated) | ~0.71 |

Post-hoc linear calibration (slope 0.977, intercept 0.834, fitted on the training split only)
corrects a systematic under-prediction bias (raw RMSE 1.03 → 0.71) without changing ranking
metrics. Raw (uncalibrated) MSE was 1.035.

### Experiment Comparison

| Experiment | Fusion | Hardware | Epochs | CI | Pearson | Spearman | MSE |
|-----------|--------|----------|--------|-----|---------|----------|-----|
| Baseline (AA-composition) | Concat | CPU (2 cores) | 17 | 0.709 | 0.411 | 0.386 | 0.931 |
| 1D-CNN protein encoder | Concat | CPU (2 cores) | 17 | 0.765 | 0.508 | 0.484 | 0.607 |
| 1D-CNN + full convergence | Concat | CPU (2 cores) | 16 | 0.772 | 0.540 | 0.497 | 0.693 |
| Attention fusion tested | Attention | CPU (2 cores) | 9 | 0.647 | 0.290 | 0.274 | 0.717 |
| **GPU convergence (best)** | **Concat** | **NVIDIA T4 GPU** | **33** | **0.793** | **0.606** | **0.534** | **1.035** |

### Key Findings

- **1D-CNN protein encoder** (+5.7% CI) outperforms amino-acid composition by preserving sequence order and local motifs
- **Concat fusion** outperforms cross-attention on this dataset size (29K pairs) — attention needs more data to learn meaningful interactions
- **GPU convergence** (100 epochs) improves ranking metrics: CI 0.772 → 0.793, Pearson 0.540 → 0.606
- **MSE vs ranking tradeoff:** GPU model ranks pairs better (important for screening) but has higher absolute error (MSE 1.035 vs 0.693)
- **Published SOTA** on Davis: ~0.88 CI (with pretrained protein embeddings + k-fold CV) — our verified 0.793 CI is competitive for a single-split, no-pretraining model

## Quick Start

### Option A: Use Pre-trained Model (recommended)

```bash
git clone https://github.com/marine9056/drug-target-Predictor.git
cd drug-target-Predictor
pip install -r requirements.txt

# Install PyG (match your PyTorch version)
pip install torch-geometric

streamlit run app/streamlit_app.py
```

### Option B: Train from Scratch

```bash
# CPU (slow, ~6 hrs)
python src/train.py --config configs/default.yaml

# GPU via Google Colab (fast, ~15 min)
# Upload notebooks/kaggle_train.ipynb to Colab with T4 GPU enabled
```

## Project Structure

```
drug-target-predictor/
├── src/                     # Source code
│   ├── data_loader.py       # Davis dataset loading
│   ├── featurization.py     # Molecular graphs & protein encoding
│   ├── model.py             # GNN model architecture
│   ├── train.py             # Training pipeline (with resume support)
│   ├── evaluate.py          # Evaluation metrics (CI, Pearson, etc.)
│   └── predict.py           # Inference API
├── app/
│   └── streamlit_app.py     # Interactive web interface
├── configs/
│   └── default.yaml         # Model & training configuration
├── notebooks/
│   ├── demo.ipynb           # Demo notebook with full pipeline
│   └── kaggle_train.ipynb   # GPU training notebook (Colab/Kaggle)
├── .github/workflows/
│   └── tests.yml            # CI: pytest on every push
├── tests/                   # Unit tests (28 tests, all passing)
├── train_gpu.py             # Standalone GPU training script
├── Dockerfile               # Container config for deployment
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
└── LICENSE                  # MIT License
```

## Tech Stack

- **Deep Learning**: PyTorch, PyTorch Geometric (GAT layers)
- **Chemistry**: RDKit (SMILES parsing, molecular graphs)
- **Protein Encoding**: 1D-CNN over integer-encoded sequences
- **Web App**: Streamlit + Plotly
- **Deployment**: Streamlit Community Cloud
- **Training**: NVIDIA T4 GPU (Google Colab), CPU fallback with checkpoint resume
- **CI/CD**: GitHub Actions (pytest on push)
- **Dataset**: Davis Kinase (MoleculeNet benchmark)

## Known Limitations

- **Absolute accuracy:** With calibration, RMSE is ~0.71 pKd units (MSE ~0.51). Individual predictions can still be off by 1–2 pKd units; ranking (CI 0.793) is more reliable than absolute values.
- **High-pKd bias:** The model under-predicts the strongest binders (pKd > 9.0) even after calibration, because the Davis dataset has few very-high-affinity pairs. A pKd of ~8.5+ reliably indicates strong binding, but the exact magnitude for the strongest binders is under-estimated.
- **Single dataset:** Trained only on Davis Kinase. Generalization to other target families (GPCRs, proteases) has not been tested.
- **No adversarial robustness testing:** Predictions have not been validated against adversarial perturbations.

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Haseeb Ur Rehman** - Bioinformatician & AI Researcher
