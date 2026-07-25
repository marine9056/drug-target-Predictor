"""
GPU Training Script for Kaggle / Google Colab
==============================================
Upload this to Kaggle or Colab, attach GPU, run all cells.

Kaggle: Free T4 GPU, ~30 hrs/week quota, no credit card
Colab: Free T4 GPU, ~12hr session limit

Usage:
  1. Upload your project as a zip to Kaggle/Colab
  2. !unzip drug-target-predictor.zip
  3. %cd drug-target-predictor
  4. !pip install -r requirements.txt
  5. Run this script
"""

import time
import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from src.model import DrugTargetPredictor
from src.data_loader import DavisDataset
from src.featurization import DrugFeaturizer, ProteinFeaturizer
from src.train import DrugTargetDataset, collate_fn, Trainer
from src.evaluate import calculate_metrics


def main():
    print("=" * 60)
    print("  DRUG-TARGET INTERACTION PREDICTOR — GPU TRAINING")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {device_name}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No GPU detected! Using CPU.")
    print()

    # Load config
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    # Override for GPU training
    config["training"]["epochs"] = 100
    config["hardware"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["hardware"]["num_workers"] = 2 if torch.cuda.is_available() else 0

    # Load data
    print("Loading dataset...")
    dataset = DavisDataset(data_dir=config["paths"]["data_dir"])
    df = dataset.load(force_download=False)
    train_df, val_df, test_df = dataset.split_data(df)

    drug_feat = DrugFeaturizer(max_atoms=config["data"]["max_drug_atoms"])
    prot_feat = ProteinFeaturizer(max_length=config["data"]["max_protein_length"])

    train_dataset = DrugTargetDataset(train_df, drug_feat, prot_feat)
    val_dataset = DrugTargetDataset(val_df, drug_feat, prot_feat)
    test_dataset = DrugTargetDataset(test_df, drug_feat, prot_feat)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=config["hardware"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=config["hardware"]["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=config["hardware"]["num_workers"])

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build model
    model = DrugTargetPredictor(
        drug_encoder_config=config["model"]["drug_encoder"],
        protein_encoder_config=config["model"]["protein_encoder"],
        fusion_type=config["model"]["fusion"]["type"],
        fusion_dim=config["model"]["fusion"]["hidden_dim"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters")
    print(f"  Fusion: {config['model']['fusion']['type']}")
    print()

    # Train
    print("Starting training...")
    print("-" * 60)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config["hardware"]["device"],
    )

    start = time.time()
    trainer.train(num_epochs=config["training"]["epochs"])
    elapsed = time.time() - start

    print(f"\nTraining finished in {elapsed / 60:.1f} minutes")
    print(f"Best val loss: {trainer.best_loss:.4f}")
    print()

    # Evaluate on test set
    print("=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch["drug_data"].to(trainer.device), batch["protein"].to(trainer.device))
            all_preds.extend(pred.cpu().squeeze().tolist())
            all_labels.extend(batch["target"].squeeze().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_labels, all_preds)

    print()
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*40}")
    for k, v in metrics.items():
        if "pvalue" not in k:
            print(f"  {k:<30} {v:>10.4f}")
    print()

    # Save results
    results_path = Path("outputs/gpu_results.txt")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        f.write("GPU Training Results\n")
        f.write(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"Training time: {elapsed / 60:.1f} min\n")
        f.write(f"Best val loss: {trainer.best_loss:.4f}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Results saved to {results_path}")
    print("Done! Download best_model.pt from models/checkpoints/")


if __name__ == "__main__":
    main()
