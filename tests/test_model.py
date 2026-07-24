"""
Test Model Module
================
Unit tests for model architecture.
"""

import pytest
import torch
import numpy as np

from src.model import DrugEncoder, ProteinEncoder, DrugTargetPredictor


class TestDrugEncoder:
    """Tests for DrugEncoder (GAT) class."""

    def test_initialization(self):
        encoder = DrugEncoder(input_dim=136, hidden_dim=64, num_layers=2, heads=2)
        assert encoder.num_layers == 2

    def test_forward_pass(self):
        encoder = DrugEncoder(input_dim=136, hidden_dim=64, num_layers=2, heads=2)

        x = torch.randn(20, 136)
        edge_index = torch.randint(0, 20, (2, 30))
        batch = torch.zeros(20, dtype=torch.long)
        batch[10:] = 1

        output = encoder(x, edge_index, batch)
        assert output.shape == (2, 64)

    def test_different_gnn_types(self):
        for gnn_type in ["gat", "gcn"]:
            encoder = DrugEncoder(input_dim=136, hidden_dim=64, num_layers=2, heads=2, gnn_type=gnn_type)
            x = torch.randn(30, 136)
            edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9]])
            batch = torch.zeros(30, dtype=torch.long)
            batch[:15] = 0
            batch[15:] = 1
            output = encoder(x, edge_index, batch)
            assert output.shape[1] == 64


class TestProteinEncoder:
    """Tests for ProteinEncoder (1D-CNN) class."""

    def test_initialization(self):
        encoder = ProteinEncoder(vocab_size=22, hidden_dim=128)
        assert encoder is not None

    def test_forward_pass(self):
        encoder = ProteinEncoder(vocab_size=22, hidden_dim=128)

        x = torch.randint(0, 21, (4, 500))
        output = encoder(x)

        assert output.shape == (4, 128)

    def test_different_sequence_lengths(self):
        encoder = ProteinEncoder(vocab_size=22, hidden_dim=128)

        for seq_len in [50, 200, 500, 1000]:
            x = torch.randint(0, 21, (2, seq_len))
            output = encoder(x)
            assert output.shape == (2, 128), f"Failed for seq_len={seq_len}"

    def test_padding_token(self):
        encoder = ProteinEncoder(vocab_size=22, hidden_dim=128)

        x = torch.full((2, 500), 21, dtype=torch.long)
        output = encoder(x)
        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()


class TestDrugTargetPredictor:
    """Tests for main prediction model."""

    def test_initialization(self):
        model = DrugTargetPredictor(
            drug_encoder_config={"input_dim": 136, "hidden_dim": 64, "num_layers": 2, "heads": 2},
            protein_encoder_config={"vocab_size": 22, "hidden_dim": 64},
            fusion_dim=128
        )
        assert model is not None

    def test_parameter_count(self):
        model = DrugTargetPredictor()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_model_save_load(self, tmp_path):
        model = DrugTargetPredictor(
            drug_encoder_config={"input_dim": 136, "hidden_dim": 64, "num_layers": 2, "heads": 2},
            protein_encoder_config={"vocab_size": 22, "hidden_dim": 64}
        )

        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        loaded = DrugTargetPredictor(
            drug_encoder_config={"input_dim": 136, "hidden_dim": 64, "num_layers": 2, "heads": 2},
            protein_encoder_config={"vocab_size": 22, "hidden_dim": 64}
        )
        loaded.load_state_dict(torch.load(save_path, weights_only=True))

        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
