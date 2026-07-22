"""
Training Module
===============
Training pipeline for drug-target interaction prediction.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugTargetDataset(Dataset):
    """
    Dataset for drug-target pairs.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        drug_featurizer,
        protein_featurizer,
        max_atoms: int = 100,
        max_length: int = 1000
    ):
        """
        Args:
            data: DataFrame with drug_smiles, protein_sequence, kd_value
            drug_featurizer: Drug featurizer instance
            protein_featurizer: Protein featurizer instance
            max_atoms: Maximum atoms per molecule
            max_length: Maximum protein sequence length
        """
        self.data = data.reset_index(drop=True)
        self.drug_featurizer = drug_featurizer
        self.protein_featurizer = protein_featurizer
        self.max_atoms = max_atoms
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get features
        smiles = row["drug_smiles"]
        sequence = row["protein_sequence"]
        kd_value = row["kd_value"]
        
        # Process drug
        try:
            graph = self.drug_featurizer.smiles_to_graph(smiles)
            drug_x = torch.tensor(graph.node_features, dtype=torch.float)
            drug_edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
            drug_edge_attr = torch.tensor(graph.edge_attr, dtype=torch.float) if graph.edge_attr is not None else None
        except Exception as e:
            logger.warning(f"Error processing drug {smiles}: {e}")
            drug_x = torch.zeros((self.max_atoms, 136), dtype=torch.float)
            drug_edge_index = torch.zeros((2, 0), dtype=torch.long)
            drug_edge_attr = None
        
        # Process protein
        protein_encoding = self.protein_featurizer.sequence_to_encoding(sequence)
        protein_tensor = torch.tensor(protein_encoding, dtype=torch.long)
        
        # Target
        target = torch.tensor([kd_value], dtype=torch.float)
        
        return {
            "drug_x": drug_x,
            "drug_edge_index": drug_edge_index,
            "drug_edge_attr": drug_edge_attr,
            "protein": protein_tensor,
            "target": target,
            "smiles": smiles,
            "sequence": sequence
        }


def collate_fn(batch):
    """
    Custom collate function for batching drug-target pairs.
    """
    from torch_geometric.data import Data, Batch
    
    drug_x_list = []
    drug_edge_index_list = []
    drug_edge_attr_list = []
    protein_list = []
    target_list = []
    
    for item in batch:
        drug_x_list.append(item["drug_x"])
        drug_edge_index_list.append(item["drug_edge_index"])
        if item["drug_edge_attr"] is not None:
            drug_edge_attr_list.append(item["drug_edge_attr"])
        protein_list.append(item["protein"])
        target_list.append(item["target"])
    
    # Create drug batch
    drug_data_list = []
    for i in range(len(drug_x_list)):
        data = Data(
            x=drug_x_list[i],
            edge_index=drug_edge_index_list[i],
            edge_attr=drug_edge_attr_list[i] if drug_edge_attr_list else None
        )
        drug_data_list.append(data)
    
    drug_batch = Batch.from_data_list(drug_data_list)
    
    # Stack proteins and targets
    protein_batch = torch.stack(protein_list)
    target_batch = torch.stack(target_list)
    
    return {
        "drug_data": drug_batch,
        "protein": protein_batch,
        "target": target_batch
    }


class Trainer:
    """
    Training pipeline for drug-target interaction models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "auto"
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device (auto, cuda, cpu)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Loss function
        loss_type = config.get("loss", {}).get("type", "mse")
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.HuberLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("training", {}).get("learning_rate", 0.001),
            weight_decay=config.get("training", {}).get("weight_decay", 0.0001)
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.get("training", {}).get("scheduler_factor", 0.5),
            patience=config.get("training", {}).get("scheduler_patience", 5),
            min_lr=config.get("training", {}).get("min_lr", 0.00001)
        )
        
        # Early stopping
        self.patience = config.get("training", {}).get("patience", 15)
        self.counter = 0
        self.best_loss = float("inf")
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("paths", {}).get("checkpoint_dir", "models/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            drug_data = batch["drug_data"].to(self.device)
            protein = batch["protein"].to(self.device)
            target = batch["target"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            prediction = self.model(drug_data, protein)
            loss = self.criterion(prediction, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if (batch_idx + 1) % self.config.get("logging", {}).get("log_interval", 10) == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: Loss = {loss.item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                drug_data = batch["drug_data"].to(self.device)
                protein = batch["protein"].to(self.device)
                target = batch["target"].to(self.device)
                
                prediction = self.model(drug_data, protein)
                loss = self.criterion(prediction, target)
                
                total_loss += loss.item()
                predictions.extend(prediction.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        return avg_loss, mse, mae
    
    def train(self, num_epochs: int = 100):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of training epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_mse, val_mae = self.validate()
            logger.info(f"Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
                self.save_checkpoint(epoch, val_loss)
                logger.info("  ✓ New best model saved!")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        logger.info(f"\nTraining complete! Best validation loss: {self.best_loss:.4f}")
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }
        
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load model checkpoint."""
        if path is None:
            path = self.checkpoint_dir / "best_model.pt"
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch'] + 1})")
        
        return checkpoint


def train_model(config: dict):
    """
    Main training function.
    
    Args:
        config: Training configuration
    """
    from src.data_loader import DavisDataset
    from src.featurization import DrugFeaturizer, ProteinFeaturizer
    from src.model import DrugTargetPredictor
    
    # Load data
    logger.info("Loading dataset...")
    dataset = DavisDataset(data_dir=config.get("paths", {}).get("data_dir", "data"))
    df = dataset.load()
    train_df, val_df, test_df = dataset.split_data(
        df,
        test_size=config.get("data", {}).get("test_size", 0.2),
        val_size=config.get("data", {}).get("val_size", 0.1)
    )
    
    # Create featurizers
    drug_featurizer = DrugFeaturizer(
        max_atoms=config.get("data", {}).get("max_drug_atoms", 100)
    )
    protein_featurizer = ProteinFeaturizer(
        max_length=config.get("data", {}).get("max_protein_length", 1000)
    )
    
    # Create datasets
    train_dataset = DrugTargetDataset(train_df, drug_featurizer, protein_featurizer)
    val_dataset = DrugTargetDataset(val_df, drug_featurizer, protein_featurizer)
    
    # Create data loaders
    batch_size = config.get("training", {}).get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get("hardware", {}).get("num_workers", 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get("hardware", {}).get("num_workers", 0)
    )
    
    # Build model
    model = DrugTargetPredictor(
        drug_encoder_config=config.get("model", {}).get("drug_encoder"),
        protein_encoder_config=config.get("model", {}).get("protein_encoder"),
        fusion_type=config.get("model", {}).get("fusion", {}).get("type", "concat"),
        fusion_dim=config.get("model", {}).get("fusion", {}).get("hidden_dim", 256)
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.get("hardware", {}).get("device", "auto")
    )
    
    num_epochs = config.get("training", {}).get("epochs", 100)
    trainer.train(num_epochs=num_epochs)
    
    return model, trainer


# Main entry point
if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Train drug-target interaction model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    
    # Train
    model, trainer = train_model(config)
    
    print("\nTraining complete!")
