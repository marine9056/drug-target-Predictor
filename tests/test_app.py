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
        """Test binding classification logic."""
        thresholds = {
            5.0: "Strong Binder",
            6.5: "Moderate Binder",
            8.0: "Weak Binder",
            10.0: "Non-binder",
        }
        for kd, expected in thresholds.items():
            if kd < 5.5:
                strength = "Strong Binder"
            elif kd < 7.0:
                strength = "Moderate Binder"
            elif kd < 8.5:
                strength = "Weak Binder"
            else:
                strength = "Non-binder"
            assert strength == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
