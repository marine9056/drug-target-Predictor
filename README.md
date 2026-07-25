# Drug-Target Interaction Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/pyg-2.4+-ff6f00.svg)](https://pyg.org/)

A Graph Neural Network model that predicts binding affinity between drug molecules and protein targets, trained on the **Davis Kinase Dataset** (29,444 drug-target pairs).

## Live Demo

> **Status:** HuggingFace Spaces deployment is in progress. The app runs locally via `streamlit run app/streamlit_app.py`.

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

### Verified Results (CPU, local checkpoint)

Trained on Davis Kinase Dataset (29,444 pairs, 80/10/10 split, seed=42):

| Metric | Value |
|--------|-------|
| MSE | 0.607 |
| MAE | 0.489 |
| R² | 0.420 |
| Concordance Index (CI) | **0.772** |
| Pearson Correlation | **0.540** |
| Spearman Correlation | **0.497** |

### GPU Training (Colab T4, 100 epochs — output metrics)

Training on NVIDIA T4 (Google Colab) showed improved ranking:
CI 0.846, Pearson 0.912, Spearman 0.906, but MSE 1.451.

> **Note:** The GPU checkpoint needs re-downloading from Colab to locally verify
> and reproduce these results. The Colab session expired before the checkpoint
> was saved. Retraining is in progress.

### Experiment Comparison

| Experiment | Fusion | Hardware | Epochs | CI | Pearson | Spearman |
|-----------|--------|----------|--------|-----|---------|----------|
| Baseline (AA-composition) | Concat | CPU (2 cores) | 17 | 0.709 | 0.411 | 0.386 |
| 1D-CNN protein encoder | Concat | CPU (2 cores) | 17 | 0.765 | 0.508 | 0.484 |
| 1D-CNN + more training | Concat | CPU (2 cores) | 27 | **0.772** | **0.540** | **0.497** |
| Attention fusion tested | Attention | CPU (2 cores) | 9 | 0.647 | 0.290 | 0.274 |
| GPU convergence (output) | Concat | NVIDIA T4 GPU | 100 | 0.846 | 0.912 | 0.906 |

### Key Findings

- **1D-CNN protein encoder** (+5.7% CI) outperforms amino-acid composition by preserving sequence order and local motifs
- **Concat fusion** outperforms cross-attention on this dataset size (29K pairs) — attention needs more data to learn meaningful interactions
- **GPU convergence** (100 epochs) dramatically improves ranking metrics over CPU early-stopping (27 epochs)
- **Published SOTA** on Davis: ~0.88 CI — our verified 0.772 CI is competitive; GPU result of 0.846 is pending local verification

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
- **Deployment**: Docker on HuggingFace Spaces
- **Training**: NVIDIA T4 GPU (Google Colab), CPU fallback with checkpoint resume
- **CI/CD**: GitHub Actions (pytest on push)
- **Dataset**: Davis Kinase (MoleculeNet benchmark)

## Known Limitations

- **MSE vs ranking tradeoff:** The model compresses predictions into a narrow range (4.3–8.0) while true labels span 5.0–10.4. This hurts MSE (0.607) but preserves ranking order (CI 0.772, Spearman 0.497), which is more useful for drug screening.
- **High-pKd bias:** The model under-predicts high-affinity pairs (pKd > 9.0). This is a known limitation of the Davis dataset distribution — only a small fraction of pairs have very high binding.
- **Single dataset:** Trained only on Davis Kinase. Generalization to other target families (GPCRs, proteases) has not been tested.
- **CPU-only validated:** Full validation was done on CPU (2 cores, 27 epochs). GPU checkpoint was lost from Colab session — retraining in progress.
- **No adversarial robustness testing:** Predictions have not been validated against adversarial perturbations.

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Haseeb Ur Rehman** - Bioinformatician & AI Researcher
