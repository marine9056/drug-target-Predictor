"""
Prediction Module
=================
Inference utilities for predicting drug-target binding.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import Dict, List, Optional, Union

from src.featurization import DrugFeaturizer, ProteinFeaturizer
from src.model import DrugTargetPredictor


class BindingPredictor:
    """
    Predicts drug-target binding affinity.
    """
    
    def __init__(
        self,
        model: DrugTargetPredictor,
        device: str = "auto",
        max_atoms: int = 100,
        max_length: int = 1000
    ):
        """
        Args:
            model: Trained model
            device: Device for inference
            max_atoms: Maximum atoms per molecule
            max_length: Maximum protein sequence length
        """
        self.model = model
        self.max_atoms = max_atoms
        self.max_length = max_length
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Featurizers
        self.drug_featurizer = DrugFeaturizer(max_atoms=max_atoms)
        self.protein_featurizer = ProteinFeaturizer(max_length=max_length)
        
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto"
    ) -> 'BindingPredictor':
        """
        Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            
        Returns:
            BindingPredictor instance
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Rebuild model from config
        config = checkpoint.get("config", {})
        model = DrugTargetPredictor(
            drug_encoder_config=config.get("model", {}).get("drug_encoder"),
            protein_encoder_config=config.get("model", {}).get("protein_encoder"),
            fusion_type=config.get("model", {}).get("fusion", {}).get("type", "concat"),
            fusion_dim=config.get("model", {}).get("fusion", {}).get("hidden_dim", 256)
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(model, device=device)
    
    def predict(
        self,
        drug_smiles: str,
        protein_sequence: str
    ) -> Dict[str, Union[float, str]]:
        """
        Predict binding affinity for a drug-target pair.
        
        Args:
            drug_smiles: SMILES string of the drug
            protein_sequence: Amino acid sequence of the protein
            
        Returns:
            Dictionary with prediction results
        """
        from torch_geometric.data import Data, Batch
        
        # Process drug
        try:
            graph = self.drug_featurizer.smiles_to_graph(drug_smiles)
            drug_data = Data(
                x=torch.tensor(graph.node_features, dtype=torch.float),
                edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
                edge_attr=torch.tensor(graph.edge_attr, dtype=torch.float) if graph.edge_attr is not None else None
            )
            drug_batch = Batch.from_data_list([drug_data])
        except Exception as e:
            return {"error": f"Invalid drug SMILES: {str(e)}"}
        
        # Process protein
        protein_encoding = self.protein_featurizer.sequence_to_encoding(protein_sequence)
        protein_tensor = torch.tensor([protein_encoding], dtype=torch.long)
        
        # Move to device
        drug_batch = drug_batch.to(self.device)
        protein_tensor = protein_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(drug_batch, protein_tensor)
        
        kd_value = prediction.cpu().numpy().item()
        
        # Classify binding strength
        binding_strength = self._classify_binding(kd_value)
        
        return {
            "kd_value": kd_value,
            "kd_units": "log(nM)",
            "binding_strength": binding_strength,
            "drug_smiles": drug_smiles,
            "protein_length": len(protein_sequence)
        }
    
    def predict_batch(
        self,
        drug_smiles_list: List[str],
        protein_sequence_list: List[str]
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Predict binding for multiple drug-target pairs.
        
        Args:
            drug_smiles_list: List of drug SMILES
            protein_sequence_list: List of protein sequences
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for smiles, seq in zip(drug_smiles_list, protein_sequence_list):
            result = self.predict(smiles, seq)
            results.append(result)
        
        return results
    
    def _classify_binding(self, kd_value: float) -> str:
        """
        Classify binding strength based on Kd value.
        
        Args:
            kd_value: Binding affinity (log Kd)
            
        Returns:
            Binding strength classification
        """
        # Kd values are in log scale
        # Lower Kd = stronger binding
        if kd_value < 5.5:
            return "Strong Binder"
        elif kd_value < 7.0:
            return "Moderate Binder"
        elif kd_value < 8.5:
            return "Weak Binder"
        else:
            return "Non-binder"
    
    def explain_prediction(
        self,
        drug_smiles: str,
        protein_sequence: str
    ) -> Dict:
        """
        Generate explanation for a prediction.
        
        Args:
            drug_smiles: Drug SMILES string
            protein_sequence: Protein sequence
            
        Returns:
            Dictionary with explanation
        """
        from src.featurization import calculate_molecular_descriptors
        
        # Get prediction
        prediction = self.predict(drug_smiles, protein_sequence)
        
        # Get molecular descriptors
        descriptors = calculate_molecular_descriptors(drug_smiles)
        
        # Protein statistics
        protein_stats = {
            "length": len(protein_sequence),
            "composition": self.protein_featurizer.sequence_to_composition(protein_sequence).tolist()
        }
        
        explanation = {
            "prediction": prediction,
            "drug_properties": descriptors,
            "protein_properties": protein_stats,
            "interpretation": self._generate_interpretation(descriptors, prediction)
        }
        
        return explanation
    
    def _generate_interpretation(
        self,
        descriptors: Dict,
        prediction: Dict
    ) -> str:
        """
        Generate human-readable interpretation.
        """
        kd = prediction.get("kd_value", 0)
        strength = prediction.get("binding_strength", "Unknown")
        
        interpretation = f"Predicted binding affinity: {kd:.2f} log(nM) ({strength})\n"
        
        if descriptors:
            mw = descriptors.get("molecular_weight", 0)
            logp = descriptors.get("logp", 0)
            
            interpretation += f"\nDrug Properties:\n"
            interpretation += f"  - Molecular Weight: {mw:.1f} g/mol\n"
            interpretation += f"  - LogP: {logp:.2f}\n"
            
            # Lipinski's Rule of Five
            hbd = descriptors.get("hbd", 0)
            hba = descriptors.get("hba", 0)
            
            lipinski_violations = 0
            if mw > 500: lipinski_violations += 1
            if logp > 5: lipinski_violations += 1
            if hbd > 5: lipinski_violations += 1
            if hba > 10: lipinski_violations += 1
            
            interpretation += f"  - Lipinski Violations: {lipinski_violations}/4\n"
        
        return interpretation


# Main entry point
if __name__ == "__main__":
    print("Prediction module loaded successfully!")
    print("Use BindingPredictor to make predictions.")
