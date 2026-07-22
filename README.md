# Drug-Target Interaction Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/pyg-2.4+-ff6f00.svg)](https://pyg.org/)

A Graph Neural Network model that predicts binding affinity between drug molecules and protein targets, trained on the **Davis Kinase Dataset** (29,444 drug-target pairs).

## Overview

Drug-target interaction prediction is a core problem in computational drug discovery. This project uses:

- **Graph Attention Networks (GAT)** to encode drug molecules from SMILES strings as molecular graphs
- **Amino acid composition encoding** for protein sequence representation
- **Concatenation-based fusion** to combine drug and protein embeddings
- **MLP prediction head** to output binding affinity (pKd values)

### Model Architecture

```
Drug (SMILES) -> Molecular Graph -> GAT Layers -> Drug Embedding (128-d)
                                                          |
                                                     Concatenate
                                                          |
Protein (Sequence) -> AA Composition -> MLP -> Protein Embedding (128-d)
                                                          |
                                                     MLP Head -> pKd
```

## Results

Trained on Davis Kinase Dataset (29,444 pairs, 80/10/10 split):

| Metric | Value |
|--------|-------|
| MSE | 0.931 |
| MAE | 0.598 |
| RMSE | 0.965 |
| Concordance Index (CI) | 0.709 |
| Pearson Correlation | 0.411 |
| Spearman Correlation | 0.386 |

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/marine9056/drug-target-predictor.git
cd drug-target-predictor
```

### 2. Create Environment

```bash
conda create -n drug-target python=3.10
conda activate drug-target
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Data

The Davis dataset will be automatically downloaded to `data/raw/` on first training run. You can also download it manually:

```bash
python src/data_loader.py
```

### 5. Train Model

```bash
python src/train.py --config configs/default.yaml
```

### 6. Run Web App

```bash
streamlit run app/streamlit_app.py
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
├── tests/                   # Unit tests
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
└── LICENSE                  # MIT License
```

## Tech Stack

- **Deep Learning**: PyTorch, PyTorch Geometric (GAT layers)
- **Chemistry**: RDKit (SMILES parsing, molecular graphs)
- **Web App**: Streamlit + Plotly
- **Dataset**: Davis Kinase (MoleculeNet benchmark)

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Haseeb Ur Rehman** - Bioinformatician & AI Researcher
