import json
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import ks_2samp
import pickle


def detect_data_drift(current_data, training_data, threshold=0.05):
    """
    Compare current data with training data
    threshold: p-value < 0.05 means drift detected
    """

    drift_report = {"timestamp": datetime.now().isoformat(), "features_with_drift": []}

    # For each numerical feature
    numerical_features = ["SeniorCitizen", "tenure", "MonthlyCharges"]

    for feature in numerical_features:
        # Get distributions
        current_dist = current_data[feature].values
        training_dist = training_data[feature].values

        # Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(current_dist, training_dist)

        # If p-value < 0.05, distributions are different
        if p_value < threshold:
            drift_report["features_with_drift"].append({
                "feature": feature,
                "p_value": f"{p_value:.4f}",
                "status": "DRIFT DETECTED 🚨",
            })

    return drift_report


# Usage:
if __name__ == "__main__":
    # Load training data
    X_train = pd.read_csv("data/processed/X_train.csv")

    # Load recent predictions
    recent_data = pd.read_csv("data/processed/recent_predictions.csv")

    # Check for drift
    report = detect_data_drift(recent_data, X_train)

    print(json.dumps(report, indent=2))
