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
        dataset = DavisDataset(data_dir=str(tmp_path))
        assert str(dataset.data_dir) == str(tmp_path)

    def test_create_curated_data(self, tmp_path):
        dataset = DavisDataset(data_dir=str(tmp_path))
        df = dataset._create_curated_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "drug_smiles" in df.columns
        assert "protein_sequence" in df.columns
        assert "kd_value" in df.columns
        assert df["kd_value"].between(2.0, 11.0).all()

    def test_split_data(self, tmp_path):
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
        dataset = DavisDataset(data_dir=str(tmp_path))

        df = dataset._create_curated_data()
        df.to_csv(tmp_path / "davis_raw.csv", index=False)

        loaded_df = dataset.load(force_download=False)
        assert len(loaded_df) == len(df)


class TestDrugFeaturizer:
    """Tests for drug molecular graph featurization."""

    def test_valid_smiles(self):
        from src.featurization import DrugFeaturizer

        featurizer = DrugFeaturizer(max_atoms=50)
        graph = featurizer.smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")

        assert graph.node_features.shape == (50, 136)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_attr.shape[1] == 6

    def test_invalid_smiles_raises(self):
        from src.featurization import DrugFeaturizer

        featurizer = DrugFeaturizer()
        with pytest.raises(ValueError):
            featurizer.smiles_to_graph("not_a_smiles")

    def test_empty_smiles_raises(self):
        from src.featurization import DrugFeaturizer

        featurizer = DrugFeaturizer()
        with pytest.raises((ValueError, IndexError)):
            featurizer.smiles_to_graph("")

    def test_large_molecule_truncated(self):
        from src.featurization import DrugFeaturizer

        featurizer = DrugFeaturizer(max_atoms=10)
        # Aspirin has 13 heavy atoms, should be truncated to 10
        graph = featurizer.smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")
        assert graph.node_features.shape[0] == 10

    def test_feature_dimensions(self):
        from src.featurization import DrugFeaturizer

        featurizer = DrugFeaturizer(max_atoms=20)
        graph = featurizer.smiles_to_graph("CCO")

        assert graph.node_features.shape == (20, 136)
        assert graph.edge_index.shape[0] == 2


class TestProteinFeaturizer:
    """Tests for protein sequence featurization."""

    def test_encoding(self):
        from src.featurization import ProteinFeaturizer

        featurizer = ProteinFeaturizer(max_length=100)
        encoding = featurizer.sequence_to_encoding("MKTLLLTLVVVTIVCLDL")

        assert len(encoding) == 100
        assert all(0 <= e <= 21 for e in encoding)

    def test_composition(self):
        from src.featurization import ProteinFeaturizer

        featurizer = ProteinFeaturizer()
        comp = featurizer.sequence_to_composition("AAAA")

        assert comp.shape == (21,)
        assert comp[0] == 1.0  # All alanine (index 0)
        assert comp.sum() == pytest.approx(1.0)

    def test_empty_sequence(self):
        from src.featurization import ProteinFeaturizer

        featurizer = ProteinFeaturizer(max_length=50)
        encoding = featurizer.sequence_to_encoding("")

        assert len(encoding) == 50
        assert all(e == 21 for e in encoding)  # All padding

    def test_long_sequence_truncated(self):
        from src.featurization import ProteinFeaturizer

        featurizer = ProteinFeaturizer(max_length=10)
        encoding = featurizer.sequence_to_encoding("M" * 20)

        assert len(encoding) == 10

    def test_invalid_characters_filtered(self):
        from src.featurization import ProteinFeaturizer

        featurizer = ProteinFeaturizer(max_length=10)
        encoding = featurizer.sequence_to_encoding("MKT123XZ")

        assert len(encoding) == 10
        assert all(0 <= e <= 21 for e in encoding)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
