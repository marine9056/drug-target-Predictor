# Drug-Target Interaction Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/pyg-2.4+-ff6f00.svg)](https://pyg.org/)

A Graph Neural Network model that predicts binding affinity between drug molecules and protein targets, trained on the **Davis Kinase Dataset** (29,444 drug-target pairs).

## Live Demo

Try the deployed app: **[HuggingFace Spaces Demo](https://haseeb3454-drug-target-predictor.hf.space)**

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

Trained on Davis Kinase Dataset (29,444 pairs, 80/10/10 split, GPU-accelerated):

| Metric | Value |
|--------|-------|
| MSE | 1.451 |
| MAE | 0.954 |
| R² | 0.548 |
| Concordance Index (CI) | **0.846** |
| Pearson Correlation | **0.912** |
| Spearman Correlation | **0.906** |

## Quick Start

### Option A: Use Pre-trained Model (recommended)

```bash
git clone https://github.com/marine9056/drug-target-Predictor.git
cd drug-target-Predictor
pip install -r requirements.txt
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
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation metrics (CI, Pearson, etc.)
│   └── predict.py           # Inference API
├── app/
│   └── streamlit_app.py     # Interactive web interface
├── configs/
│   └── default.yaml         # Model & training configuration
├── notebooks/
│   └── demo.ipynb           # Demo notebook with full pipeline
├── tests/                   # Unit tests (28 tests, all passing)
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
- **Dataset**: Davis Kinase (MoleculeNet benchmark)

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Haseeb Ur Rehman** - Bioinformatician & AI Researcher
