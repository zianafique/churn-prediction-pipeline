"""
Tests for model predictions
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestModelLoading:
    """Test model loading"""

    def test_model_file_exists(self):
        """Test that model file exists"""
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "logistic_regression_tuned.pkl"
        )
        assert os.path.exists(model_path), f"Model file not found at {model_path}"

    def test_model_can_be_loaded(self):
        """Test that model can be loaded successfully"""
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "logistic_regression_tuned.pkl"
        )
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            assert model is not None, "Model should load successfully"
        except Exception as e:
            pytest.fail(f"Failed to load model: {str(e)}")


class TestModelPredictions:
    """Test model prediction functionality"""

    @pytest.fixture
    def model(self):
        """Load model fixture"""
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "logistic_regression_tuned.pkl"
        )
        with open(model_path, "rb") as f:
            return pickle.load(f)

    @pytest.fixture
    def test_data(self):
        """Load test data fixture"""
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
        return X_test, y_test

    def test_predictions_shape(self, model, test_data):
        """Test that predictions have correct shape"""
        X_test, _ = test_data
        predictions = model.predict(X_test)
        assert predictions.shape[0] == X_test.shape[0], (
            "Prediction count should match input count"
        )

    def test_predictions_are_yes_no(self, model, test_data):
        """Test that predictions are 'Yes' or 'No'"""
        X_test, _ = test_data
        predictions = model.predict(X_test)
        unique_preds = set(predictions)
        # YOUR DATA: Predictions should be 'Yes' and 'No', not 0/1
        assert unique_preds.issubset({"Yes", "No"}), (
            f"Predictions should be 'Yes'/'No', got {unique_preds}"
        )

    def test_probability_predictions_range(self, model, test_data):
        """Test that probability predictions are in [0, 1]"""
        X_test, _ = test_data
        probs = model.predict_proba(X_test)[:, 1]
        assert np.all(probs >= 0) and np.all(probs <= 1), (
            "Probabilities should be in [0, 1]"
        )

    def test_model_accuracy_is_reasonable(self, model, test_data):
        """Test that model accuracy is above baseline (50%)"""
        from sklearn.metrics import accuracy_score

        X_test, y_test = test_data
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.70, f"Accuracy should be > 70%, got {accuracy:.2%}"

    def test_single_prediction(self, model, test_data):
        """Test that model works with single sample"""
        X_test, _ = test_data
        single_sample = X_test.iloc[[0]]
        prediction = model.predict(single_sample)
        assert len(prediction) == 1, "Should return single prediction"

    def test_no_nan_in_predictions(self, model, test_data):
        """Test that predictions don't contain NaN or None"""
        X_test, _ = test_data
        predictions = model.predict(X_test)

        # Check for None values (works with strings)
        assert all(pred is not None for pred in predictions), (
            "Predictions should not contain None"
        )

        # Check for empty strings
        assert all(isinstance(pred, str) and len(pred) > 0 for pred in predictions), (
            "Predictions should be non-empty strings"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
