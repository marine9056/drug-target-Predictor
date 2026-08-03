"""
Test Application Module
======================
Unit tests for evaluation and app logic.
"""

import pytest
import numpy as np


class TestEvaluation:
    """Tests for evaluation metrics."""

    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        from src.evaluate import calculate_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = calculate_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "ci" in metrics

        assert metrics["r2"] > 0.9
        assert metrics["ci"] > 0.9
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0

    def test_concordance_index(self):
        """Test concordance index calculation."""
        from src.evaluate import concordance_index

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        ci = concordance_index(y_true, y_pred)
        assert ci == 1.0

        y_pred_random = np.array([3.0, 1.0, 2.0])
        ci_random = concordance_index(y_true, y_pred_random)
        assert 0.0 <= ci_random <= 1.0

    def test_app_syntax(self):
        """Test that app module has valid syntax."""
        import ast
        with open("app/streamlit_app.py", "r") as f:
            code = f.read()
        ast.parse(code)

    def test_binding_classification(self):
        """Test binding classification logic.

        pKd = -log10(Kd in M). HIGHER pKd = LOWER Kd = STRONGER binding.
        """
        from src.predict import BindingPredictor

        thresholds = {
            5.0: "Non-binder",
            6.5: "Weak Binder",
            8.0: "Moderate Binder",
            10.0: "Strong Binder",
        }
        for kd, expected in thresholds.items():
            strength = BindingPredictor._classify_binding(kd)
            assert strength == expected

    def test_calibration_application(self):
        """Test that linear calibration shifts predictions the right way."""
        from src.predict import BindingPredictor

        class FakeModel:
            def to(self, device):
                return self
            def eval(self):
                return self

        calib = {"slope": 2.0, "intercept": 1.0}
        p = BindingPredictor(model=FakeModel(), device="cpu", calibration=calib)
        assert p._apply_calibration(3.0) == 7.0
        assert p._apply_calibration(5.0) == 11.0

    def test_protein_validation(self):
        """Test that garbage/empty protein sequences are rejected."""
        from src.predict import BindingPredictor

        class FakeModel:
            def to(self, device):
                return self
            def eval(self):
                return self

        p = BindingPredictor(model=FakeModel(), device="cpu")
        assert p._valid_protein("12345 !!! hello world $$$") is False
        assert p._valid_protein("") is False
        assert p._valid_protein("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
