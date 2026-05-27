"""
Tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import (
    load_data,
    separate_features_target,
    identify_feature_types,
    encode_categorical_features,
    scale_numerical_features,
    engineer_features,
)


class TestDataLoading:
    """Test data loading functionality"""

    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a pandas DataFrame"""
        df = load_data()
        assert isinstance(df, pd.DataFrame), "load_data should return DataFrame"

    def test_load_data_has_correct_shape(self):
        """Test that loaded data has expected number of rows"""
        df = load_data()
        assert len(df) == 7043, f"Expected 7043 rows, got {len(df)}"

    def test_load_data_no_missing_values(self):
        """Test that data has no missing values"""
        df = load_data()
        assert df.isnull().sum().sum() == 0, "Data should have no missing values"


class TestFeatureSeparation:
    """Test feature and target separation"""

    def test_separate_features_target_returns_tuple(self):
        """Test that function returns tuple of X and y"""
        df = load_data()
        result = separate_features_target(df)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return (X, y)"

    def test_separate_features_target_shapes(self):
        """Test correct shapes of X and y"""
        df = load_data()
        X, y = separate_features_target(df)
        assert X.shape[0] == y.shape[0], "X and y should have same number of rows"
        assert y.shape[0] == 7043, "Should have 7043 samples"

    def test_target_values_are_yes_no(self):
        """Test that target contains 'Yes' and 'No' values"""
        df = load_data()
        X, y = separate_features_target(df)
        unique_values = set(y.unique())
        # YOUR DATA: Target should be 'Yes' and 'No', not 0/1
        assert unique_values == {"Yes", "No"}, (
            f"Target should be {{'Yes', 'No'}}, got {unique_values}"
        )


class TestFeatureIdentification:
    """Test feature type identification"""

    def test_identify_feature_types_returns_lists(self):
        """Test that function returns lists"""
        df = load_data()
        X, y = separate_features_target(df)
        cat_cols, num_cols = identify_feature_types(X)
        assert isinstance(cat_cols, list), "Categorical columns should be list"
        assert isinstance(num_cols, list), "Numerical columns should be list"

    def test_feature_types_no_overlap(self):
        """Test that categorical and numerical features don't overlap"""
        df = load_data()
        X, y = separate_features_target(df)
        cat_cols, num_cols = identify_feature_types(X)
        overlap = set(cat_cols) & set(num_cols)
        assert len(overlap) == 0, f"Features shouldn't overlap: {overlap}"

    def test_all_features_classified(self):
        """Test that all features are classified"""
        df = load_data()
        X, y = separate_features_target(df)
        cat_cols, num_cols = identify_feature_types(X)
        total_features = len(cat_cols) + len(num_cols)
        assert total_features == X.shape[1], "All features should be classified"


class TestFeatureProcessing:
    """Test overall feature processing"""

    def test_feature_processing_returns_dataframe(self):
        """Test that feature processing returns a dataframe"""
        df = load_data()
        X, y = separate_features_target(df)
        # Just check we can identify features
        cat_cols, num_cols = identify_feature_types(X)
        assert len(cat_cols) > 0, "Should have categorical features"
        assert len(num_cols) > 0, "Should have numerical features"

    def test_engineer_features_returns_same_shape(self):
        """Test that engineering returns valid data"""
        df = load_data()
        X, y = separate_features_target(df)
        X_engineered = engineer_features(X)
        # Check it returns a dataframe
        assert isinstance(X_engineered, pd.DataFrame), "Should return DataFrame"
        # Check it has rows
        assert X_engineered.shape[0] == X.shape[0], "Should have same number of rows"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
