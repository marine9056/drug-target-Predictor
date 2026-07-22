"""
Test Data Module
================
Unit tests for data loading and processing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import DavisDataset


class TestDavisDataset:
    """Tests for DavisDataset class."""

    def test_init(self, tmp_path):
        """Test dataset initialization."""
        dataset = DavisDataset(data_dir=str(tmp_path))
        assert str(dataset.data_dir) == str(tmp_path)

    def test_create_curated_data(self, tmp_path):
        """Test curated data creation."""
        dataset = DavisDataset(data_dir=str(tmp_path))
        df = dataset._create_curated_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "drug_smiles" in df.columns
        assert "protein_sequence" in df.columns
        assert "kd_value" in df.columns

    def test_split_data(self, tmp_path):
        """Test data splitting."""
        dataset = DavisDataset(data_dir=str(tmp_path))
        df = dataset._create_curated_data()

        train, val, test = dataset.split_data(df, test_size=0.2, val_size=0.1)

        total = len(train) + len(val) + len(test)
        assert total == len(df)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert abs(len(train) / len(df) - 0.7) < 0.05

    def test_load_cached(self, tmp_path):
        """Test loading cached data."""
        dataset = DavisDataset(data_dir=str(tmp_path))

        df = dataset._create_curated_data()
        df.to_csv(tmp_path / "davis_raw.csv", index=False)

        loaded_df = dataset.load(force_download=False)

        assert len(loaded_df) == len(df)


class TestFeaturization:
    """Tests for featurization module."""

    def test_drug_featurizer(self):
        """Test drug featurization."""
        from src.featurization import DrugFeaturizer

        featurizer = DrugFeaturizer(max_atoms=50)
        smiles = "CC(=O)Oc1ccccc1C(=O)O"

        graph = featurizer.smiles_to_graph(smiles)

        assert graph.node_features.shape[0] == 50
        assert graph.edge_index.shape[0] == 2
        assert graph.node_features.shape[1] == 136

    def test_protein_featurizer(self):
        """Test protein featurization."""
        from src.featurization import ProteinFeaturizer

        featurizer = ProteinFeaturizer(max_length=100)
        sequence = "MKTLLLTLVVVTIVCLDL"

        encoding = featurizer.sequence_to_encoding(sequence)

        assert len(encoding) == 100
        assert all(0 <= e <= 21 for e in encoding)

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        from src.featurization import DrugFeaturizer

        featurizer = DrugFeaturizer()

        with pytest.raises(ValueError):
            featurizer.smiles_to_graph("invalid_smiles")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
