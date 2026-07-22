"""
Featurization Module
====================
Converts drugs and proteins into numerical representations for model input.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.Chem import rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not installed. Molecular featurization will be limited.")

try:
    import torch
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: PyTorch Geometric not installed. Graph featurization will be limited.")


# Amino acid vocabulary
AMINO_ACIDS = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

# Atom features for molecular graphs
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2, 3],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ] if HAS_RDKIT else [],
    'is_aromatic': [True, False]
}


@dataclass
class MolecularGraph:
    """Container for molecular graph data."""
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: Optional[np.ndarray] = None


class DrugFeaturizer:
    """
    Converts drug SMILES strings to molecular graphs.
    """
    
    def __init__(self, max_atoms: int = 100):
        """
        Args:
            max_atoms: Maximum number of atoms in molecule
        """
        self.max_atoms = max_atoms
        
    def smiles_to_mol(self, smiles: str):
        """Convert SMILES to RDKit molecule object."""
        if not HAS_RDKIT:
            raise ImportError("RDKit is required for molecular featurization")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return mol
    
    def get_atom_features(self, atom) -> np.ndarray:
        """
        Extract features from a single atom.
        
        Args:
            atom: RDKit atom object
            
        Returns:
            Feature vector for the atom
        """
        features = []
        
        # Atomic number (one-hot encoded)
        atomic_num = atom.GetAtomicNum()
        features.extend([1 if i == atomic_num else 0 for i in range(1, 119)])
        
        # Degree
        degree = atom.GetDegree()
        features.extend([1 if i == degree else 0 for i in range(6)])
        
        # Formal charge
        charge = atom.GetFormalCharge()
        features.extend([1 if i == charge else 0 for i in range(-2, 4)])
        
        # Hybridization
        hybridization = atom.GetHybridization()
        for h in [Chem.rdchem.HybridizationType.SP,
                  Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2]:
            features.append(1 if hybridization == h else 0)
        
        # Aromaticity
        features.append(1 if atom.GetIsAromatic() else 0)
        
        return np.array(features, dtype=np.float32)
    
    def get_bond_features(self, bond) -> np.ndarray:
        """
        Extract features from a bond.
        
        Args:
            bond: RDKit bond object
            
        Returns:
            Feature vector for the bond
        """
        features = []
        
        # Bond type
        bond_type = bond.GetBondType()
        features.extend([
            1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
            1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0
        ])
        
        # Conjugation
        features.append(1 if bond.GetIsConjugated() else 0)
        
        # Ring membership
        features.append(1 if bond.IsInRing() else 0)
        
        return np.array(features, dtype=np.float32)
    
    def smiles_to_graph(self, smiles: str) -> MolecularGraph:
        """
        Convert SMILES string to molecular graph.
        
        Args:
            smiles: SMILES representation of molecule
            
        Returns:
            MolecularGraph with node and edge features
        """
        mol = self.smiles_to_mol(smiles)
        
        # Get atom features (nodes)
        atom_features = []
        for atom in mol.GetAtoms():
            if len(atom_features) < self.max_atoms:
                atom_features.append(self.get_atom_features(atom))
        
        # Pad to max_atoms if needed
        while len(atom_features) < self.max_atoms:
            atom_features.append(np.zeros_like(atom_features[0]))
        
        node_features = np.array(atom_features)
        
        # Get bond features (edges)
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            if i < self.max_atoms and j < self.max_atoms:
                edge_indices.extend([[i, j], [j, i]])  # Undirected
                bond_feat = self.get_bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])
        
        if edge_indices:
            edge_index = np.array(edge_indices).T
            edge_attr = np.array(edge_features)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 6), dtype=np.float32)
        
        return MolecularGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )


class ProteinFeaturizer:
    """
    Converts protein sequences to numerical features.
    """
    
    def __init__(self, max_length: int = 1000, use_esm: bool = False):
        """
        Args:
            max_length: Maximum sequence length
            use_esm: Whether to use ESM embeddings
        """
        self.max_length = max_length
        self.use_esm = use_esm
        self.vocab_size = len(AMINO_ACIDS)
        
    def sequence_to_encoding(self, sequence: str) -> np.ndarray:
        """
        Convert amino acid sequence to integer encoding.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Integer-encoded sequence
        """
        # Clean sequence
        sequence = sequence.upper().strip()
        sequence = ''.join(c for c in sequence if c in AMINO_ACIDS)
        
        # Encode
        encoding = []
        for aa in sequence[:self.max_length]:
            encoding.append(AMINO_ACIDS.get(aa, 20))  # 20 for unknown
        
        # Pad
        while len(encoding) < self.max_length:
            encoding.append(21)  # Padding token
        
        return np.array(encoding, dtype=np.int64)
    
    def sequence_to_onehot(self, sequence: str) -> np.ndarray:
        """
        Convert amino acid sequence to one-hot encoding.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            One-hot encoded sequence [seq_len, vocab_size]
        """
        encoding = self.sequence_to_encoding(sequence)
        
        # One-hot encode
        onehot = np.zeros((len(encoding), self.vocab_size + 1), dtype=np.float32)
        for i, idx in enumerate(encoding):
            onehot[i, idx] = 1.0
        
        return onehot
    
    def sequence_to_composition(self, sequence: str) -> np.ndarray:
        """
        Calculate amino acid composition.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Composition vector [21] (20 amino acids + unknown)
        """
        sequence = sequence.upper().strip()
        
        composition = np.zeros(self.vocab_size + 1, dtype=np.float32)
        total = len(sequence)
        
        if total == 0:
            return composition
        
        for aa in sequence:
            if aa in AMINO_ACIDS:
                composition[AMINO_ACIDS[aa]] += 1
            else:
                composition[20] += 1
        
        return composition / total


def create_drug_graph_batch(
    smiles_list: List[str],
    max_atoms: int = 100
) -> 'Batch':
    """
    Create a batch of molecular graphs.
    
    Args:
        smiles_list: List of SMILES strings
        max_atoms: Maximum atoms per molecule
        
    Returns:
        PyTorch Geometric Batch
    """
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric is required for graph batching")
    
    featurizer = DrugFeaturizer(max_atoms=max_atoms)
    
    graphs = []
    for smiles in smiles_list:
        try:
            graph = featurizer.smiles_to_graph(smiles)
            
            # Convert to PyTorch Geometric Data
            data = Data(
                x=torch.tensor(graph.node_features, dtype=torch.float),
                edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
                edge_attr=torch.tensor(graph.edge_attr, dtype=torch.float)
            )
            graphs.append(data)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            # Add empty graph
            data = Data(
                x=torch.zeros((max_atoms, 136), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 6), dtype=torch.float)
            )
            graphs.append(data)
    
    return Batch.from_data_list(graphs)


def featurize_protein_batch(
    sequences: List[str],
    max_length: int = 1000
) -> np.ndarray:
    """
    Create a batch of protein features.
    
    Args:
        sequences: List of protein sequences
        max_length: Maximum sequence length
        
    Returns:
        Batched protein features [batch_size, max_length]
    """
    featurizer = ProteinFeaturizer(max_length=max_length)
    
    features = []
    for seq in sequences:
        encoding = featurizer.sequence_to_encoding(seq)
        features.append(encoding)
    
    return np.array(features)


def calculate_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """
    Calculate molecular descriptors for a drug.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular descriptors
    """
    if not HAS_RDKIT:
        return {}
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    descriptors = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.RingCount(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
    }
    
    return descriptors


# Main entry point for testing
if __name__ == "__main__":
    # Test drug featurization
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    
    drug_feat = DrugFeaturizer()
    graph = drug_feat.smiles_to_graph(test_smiles)
    
    print(f"Drug: {test_smiles}")
    print(f"Node features shape: {graph.node_features.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    
    # Test protein featurization
    test_seq = "MKTLLLTLVVVTIVCLDLGYTENLYFQS"
    
    prot_feat = ProteinFeaturizer()
    encoding = prot_feat.sequence_to_encoding(test_seq)
    composition = prot_feat.sequence_to_composition(test_seq)
    
    print(f"\nProtein: {test_seq}")
    print(f"Encoding shape: {encoding.shape}")
    print(f"Composition shape: {composition.shape}")
