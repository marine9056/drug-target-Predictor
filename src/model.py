"""
Model Module
============
Graph Neural Network model for drug-target interaction prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
    from torch_geometric.nn import BatchNorm
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: PyTorch Geometric not installed. Using fallback.")


class DrugEncoder(nn.Module):
    """
    Graph Neural Network encoder for drug molecules.
    """
    
    def __init__(
        self,
        input_dim: int = 136,  # Atom feature dimension
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        gnn_type: str = "gat"
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            gnn_type: Type of GNN (gcn, gat, gin)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == "gat":
                conv = GATConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout
                )
            elif gnn_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            else:  # GIN
                conv = GCNConv(hidden_dim, hidden_dim)  # Simplified
            
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers with residual connections
        for i in range(self.num_layers):
            residual = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if residual.shape == x.shape:
                x = x + residual
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class ProteinEncoder(nn.Module):
    """
    Protein sequence encoder using composition + MLP.
    """
    
    def __init__(
        self,
        vocab_size: int = 22,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        composition_dim = vocab_size
        
        self.encoder = nn.Sequential(
            nn.Linear(composition_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        composition = torch.zeros(x.size(0), 22, device=x.device)
        for i in range(22):
            composition[:, i] = (x == i).float().sum(dim=1)
        
        total = composition.sum(dim=1, keepdim=True) + 1e-8
        composition = composition / total
        
        return self.encoder(composition)


class DrugTargetPredictor(nn.Module):
    """
    Main model for drug-target interaction prediction.
    """
    
    def __init__(
        self,
        drug_encoder_config: dict = None,
        protein_encoder_config: dict = None,
        fusion_type: str = "concat",
        fusion_dim: int = 256,
        prediction_hidden_dim: int = 128,
        prediction_num_layers: int = 2,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        """
        Args:
            drug_encoder_config: Configuration for drug encoder
            protein_encoder_config: Configuration for protein encoder
            fusion_type: Fusion method (concat, attention, bilinear)
            fusion_dim: Fusion layer dimension
            prediction_hidden_dim: Prediction head hidden dimension
            prediction_num_layers: Number of prediction layers
            dropout: Dropout rate
            output_dim: Output dimension (1 for regression)
        """
        super().__init__()
        
        # Default configs
        if drug_encoder_config is None:
            drug_encoder_config = {
                "input_dim": 136,
                "hidden_dim": 128,
                "num_layers": 3,
                "heads": 4,
                "dropout": 0.2,
                "gnn_type": "gat"
            }
        
        if protein_encoder_config is None:
            protein_encoder_config = {
                "vocab_size": 22,
                "hidden_dim": 128,
                "dropout": 0.2
            }
        
        # Encoders
        self.drug_encoder = DrugEncoder(**drug_encoder_config)
        self.protein_encoder = ProteinEncoder(**protein_encoder_config)
        
        # Fusion
        self.fusion_type = fusion_type
        drug_dim = drug_encoder_config["hidden_dim"]
        protein_dim = protein_encoder_config["hidden_dim"]
        
        if fusion_type == "concat":
            self.fusion = nn.Linear(drug_dim + protein_dim, fusion_dim)
        elif fusion_type == "attention":
            self.fusion = nn.MultiheadAttention(
                embed_dim=drug_dim,
                num_heads=4,
                dropout=dropout
            )
            self.fusion_proj = nn.Linear(drug_dim, fusion_dim)
        else:  # bilinear
            self.fusion = nn.Bilinear(drug_dim, protein_dim, fusion_dim)
        
        # Prediction head
        prediction_layers = []
        current_dim = fusion_dim
        
        for _ in range(prediction_num_layers - 1):
            prediction_layers.extend([
                nn.Linear(current_dim, prediction_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = prediction_hidden_dim
        
        prediction_layers.append(nn.Linear(current_dim, output_dim))
        
        self.prediction_head = nn.Sequential(*prediction_layers)
        
    def forward(self, drug_data, protein_sequences):
        """
        Forward pass.
        
        Args:
            drug_data: PyTorch Geometric Batch containing drug graphs
            protein_sequences: Protein sequences [batch_size, seq_length]
            
        Returns:
            Predicted binding affinities [batch_size, output_dim]
        """
        # Encode drug
        drug_embedding = self.drug_encoder(
            drug_data.x,
            drug_data.edge_index,
            drug_data.batch
        )
        
        # Encode protein
        protein_embedding = self.protein_encoder(protein_sequences)
        
        # Fusion
        if self.fusion_type == "concat":
            combined = torch.cat([drug_embedding, protein_embedding], dim=-1)
            fused = self.fusion(combined)
        elif self.fusion_type == "attention":
            # Use drug as query, protein as key/value
            drug_seq = drug_embedding.unsqueeze(0)
            protein_seq = protein_embedding.unsqueeze(0)
            attn_output, _ = self.fusion(drug_seq, protein_seq, protein_seq)
            fused = self.fusion_proj(attn_output.squeeze(0))
        else:  # bilinear
            fused = self.fusion(drug_embedding, protein_embedding)
        
        fused = F.relu(fused)
        
        # Prediction
        prediction = self.prediction_head(fused)
        
        return prediction


class BaselineModels:
    """
    Baseline models for comparison.
    """
    
    @staticmethod
    def random_forest_baseline():
        """Random Forest baseline model."""
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    @staticmethod
    def mlp_baseline(input_dim: int):
        """MLP baseline model."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )


# Model factory
def build_model(config: dict) -> nn.Module:
    """
    Build model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_type = config.get("model_type", "drug_target_predictor")
    
    if model_type == "drug_target_predictor":
        return DrugTargetPredictor(
            drug_encoder_config=config.get("drug_encoder"),
            protein_encoder_config=config.get("protein_encoder"),
            fusion_type=config.get("fusion", {}).get("type", "concat"),
            fusion_dim=config.get("fusion", {}).get("hidden_dim", 256),
            prediction_hidden_dim=config.get("prediction_head", {}).get("hidden_dim", 128),
            prediction_num_layers=config.get("prediction_head", {}).get("num_layers", 2),
            dropout=config.get("prediction_head", {}).get("dropout", 0.3)
        )
    elif model_type == "mlp_baseline":
        input_dim = config.get("input_dim", 1024)
        return BaselineModels.mlp_baseline(input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Main entry point for testing
if __name__ == "__main__":
    # Test model creation
    model = DrugTargetPredictor()
    
    print("Model created successfully!")
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
