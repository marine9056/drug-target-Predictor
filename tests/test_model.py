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
    """Tests for DrugEncoder class."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = DrugEncoder(
            input_dim=136,
            hidden_dim=64,
            num_layers=2,
            heads=2
        )

        assert encoder is not None
        assert encoder.num_layers == 2

    def test_forward_pass(self):
        """Test forward pass."""
        encoder = DrugEncoder(
            input_dim=136,
            hidden_dim=64,
            num_layers=2,
            heads=2
        )

        num_nodes = 20
        batch_size = 2

        x = torch.randn(num_nodes, 136)
        edge_index = torch.randint(0, num_nodes, (2, 30))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        batch[num_nodes // 2:] = 1

        output = encoder(x, edge_index, batch)

        assert output.shape == (batch_size, 64)


class TestProteinEncoder:
    """Tests for ProteinEncoder class."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ProteinEncoder(
            vocab_size=22,
            hidden_dim=128
        )

        assert encoder is not None

    def test_forward_pass(self):
        """Test forward pass."""
        encoder = ProteinEncoder(
            vocab_size=22,
            hidden_dim=128
        )

        batch_size = 4
        seq_length = 50

        x = torch.randint(0, 21, (batch_size, seq_length))
        output = encoder(x)

        assert output.shape == (batch_size, 128)


class TestDrugTargetPredictor:
    """Tests for main prediction model."""

    def test_initialization(self):
        """Test model initialization."""
        model = DrugTargetPredictor(
            drug_encoder_config={
                "input_dim": 136,
                "hidden_dim": 64,
                "num_layers": 2,
                "heads": 2
            },
            protein_encoder_config={
                "vocab_size": 22,
                "hidden_dim": 64
            },
            fusion_dim=128
        )

        assert model is not None

    def test_parameter_count(self):
        """Test that model has parameters."""
        model = DrugTargetPredictor()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params

    def test_model_save_load(self, tmp_path):
        """Test model save and load."""
        model = DrugTargetPredictor(
            drug_encoder_config={
                "input_dim": 136,
                "hidden_dim": 64,
                "num_layers": 2,
                "heads": 2
            },
            protein_encoder_config={
                "vocab_size": 22,
                "hidden_dim": 64
            }
        )

        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        loaded_model = DrugTargetPredictor(
            drug_encoder_config={
                "input_dim": 136,
                "hidden_dim": 64,
                "num_layers": 2,
                "heads": 2
            },
            protein_encoder_config={
                "vocab_size": 22,
                "hidden_dim": 64
            }
        )
        loaded_model.load_state_dict(torch.load(save_path, weights_only=True))

        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
