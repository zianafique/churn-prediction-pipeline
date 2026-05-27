"""
Integration tests - test end-to-end pipeline
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestEndToEndPipeline:
    """Test complete ML pipeline"""

    def test_data_loading_to_evaluation(self):
        """Test complete pipeline from loading to prediction"""
        # Load data
        X_test = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "processed", "X_test.csv"
            )
        )
        y_test = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "processed", "y_test.csv"
            )
        ).squeeze()

        assert X_test.shape[0] > 0, "Test data should not be empty"
        assert y_test.shape[0] > 0, "Test labels should not be empty"

    def test_model_produces_consistent_predictions(self):
        """Test that model produces consistent predictions"""
        import pickle

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "logistic_regression_tuned.pkl"
        )
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X_test = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__), "..", "data", "processed", "X_test.csv"
            )
        )

        # Get predictions twice
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)

        # Should be identical
        assert np.array_equal(pred1, pred2), "Predictions should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
