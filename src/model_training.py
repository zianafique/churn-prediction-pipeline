"""
Model training module.
Trains baseline models on churn data and evaluates their performance.
"""

from pyexpat import model

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import pickle
import warnings

warnings.filterwarnings("ignore")

# Configuration
RANDOM_STATE = 42
DATA_PATH = "C:/Users/afiqu/Desktop/Learning And Evaulating Python/churn-prediction-pipeline/data/processed/"
MODELS_PATH = "C:/Users/afiqu/Desktop/Learning And Evaulating Python/churn-prediction-pipeline/models/"
print("✅ Model training module imported successfully")


def load_processed_data():
    """
    Load train/val/test data and preprocessing objects

    Returns:
        dict: Contains X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load data
    X_train = pd.read_csv(f"{DATA_PATH}X_train.csv")
    X_val = pd.read_csv(f"{DATA_PATH}X_val.csv")
    X_test = pd.read_csv(f"{DATA_PATH}X_test.csv")

    y_train = pd.read_csv(f"{DATA_PATH}y_train.csv").squeeze()  # Convert to Series
    y_val = pd.read_csv(f"{DATA_PATH}y_val.csv").squeeze()
    y_test = pd.read_csv(f"{DATA_PATH}y_test.csv").squeeze()

    print(f"✅ Data loaded:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   X_test: {X_test.shape}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression baseline model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        dict: Model and performance metrics
    """
    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    # Create and train model
    model = LogisticRegression(
        max_iter=1000,  # Iterations to converge
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all CPU cores
    )

    model.fit(X_train, y_train)
    print("✅ Model trained")

    # Make predictions on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1

    # Calculate metrics
    metrics = {
        "model": model,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, pos_label="Yes"),
        "recall": recall_score(y_val, y_pred, pos_label="Yes"),
        "f1": f1_score(y_val, y_pred, pos_label="Yes"),
        "auc": roc_auc_score((y_val == "Yes").astype(int), y_pred_proba),
    }

    print(f"\n✅ Validation Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   ROC AUC:   {metrics['auc']:.4f}")

    return metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest baseline model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        dict: Model and performance metrics
    """
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=15,  # Tree depth limit
        min_samples_split=10,  # Minimum samples to split node
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all CPU cores
    )

    model.fit(X_train, y_train)
    print("✅ Model trained")

    # Make predictions on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    metrics = {
        "model": model,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, pos_label="Yes"),
        "recall": recall_score(y_val, y_pred, pos_label="Yes"),
        "f1": f1_score(y_val, y_pred, pos_label="Yes"),
        "auc": roc_auc_score((y_val == "Yes").astype(int), y_pred_proba),
        "feature_importance": model.feature_importances_,
    }

    print(f"\n✅ Validation Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   ROC AUC:   {metrics['auc']:.4f}")

    return metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        dict: Model and performance metrics
    """
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    # Create and train model
    model = XGBClassifier(
        n_estimators=100,  # Number of boosting rounds
        learning_rate=0.1,  # How much to adjust weights
        max_depth=6,  # Tree depth
        min_child_weight=1,  # Minimum weight in leaf
        subsample=0.8,  # % of samples to use per tree
        colsample_bytree=0.8,  # % of features per tree
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",  # Evaluation metric
    )
    # Convert labels to numeric
    y_train_encoded = (y_train == "Yes").astype(int)
    y_val_encoded = (y_val == "Yes").astype(int)

    # Train with early stopping on validation set
    model.fit(
        X_train,
        y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        verbose=False,
    )
    print("✅ Model trained")

    # Make predictions on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "model": model,
        "accuracy": accuracy_score(y_val_encoded, y_pred),
        "precision": precision_score(y_val_encoded, y_pred),
        "recall": recall_score(y_val_encoded, y_pred),
        "f1": f1_score(y_val_encoded, y_pred),
        "auc": roc_auc_score(y_val_encoded, y_pred_proba),
        "feature_importance": model.feature_importances_,
    }

    print(f"\n✅ Validation Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   ROC AUC:   {metrics['auc']:.4f}")

    return metrics


def compare_models(logistic_metrics, rf_metrics, xgb_metrics, feature_names):
    """
    Compare performance of all three models

    Args:
        logistic_metrics: Logistic Regression results
        rf_metrics: Random Forest results
        xgb_metrics: XGBoost results
        feature_names: Column names for feature importance

    Returns:
        str: Best model name
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Create comparison dataframe
    comparison = pd.DataFrame({
        "Logistic Regression": {
            "Accuracy": logistic_metrics["accuracy"],
            "Precision": logistic_metrics["precision"],
            "Recall": logistic_metrics["recall"],
            "F1 Score": logistic_metrics["f1"],
            "ROC AUC": logistic_metrics["auc"],
        },
        "Random Forest": {
            "Accuracy": rf_metrics["accuracy"],
            "Precision": rf_metrics["precision"],
            "Recall": rf_metrics["recall"],
            "F1 Score": rf_metrics["f1"],
            "ROC AUC": rf_metrics["auc"],
        },
        "XGBoost": {
            "Accuracy": xgb_metrics["accuracy"],
            "Precision": xgb_metrics["precision"],
            "Recall": xgb_metrics["recall"],
            "F1 Score": xgb_metrics["f1"],
            "ROC AUC": xgb_metrics["auc"],
        },
    })

    print("\n" + comparison.to_string())

    # Find best model by ROC AUC (best metric for imbalanced data)
    best_model_name = comparison.loc["ROC AUC"].idxmax()
    best_auc = comparison.loc["ROC AUC"].max()

    print(f"\n✅ Best Model: {best_model_name}")
    print(f"   ROC AUC: {best_auc:.4f}")

    # Feature importance comparison
    print(f"\n" + "=" * 70)
    print("FEATURE IMPORTANCE COMPARISON")
    print("=" * 70)

    # Get top 10 features from each tree-based model
    rf_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_metrics["feature_importance"],
    }).nlargest(10, "Importance")

    xgb_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": xgb_metrics["feature_importance"],
    }).nlargest(10, "Importance")

    print("\nRandom Forest Top 10 Features:")
    print(rf_importance.to_string(index=False))

    print("\nXGBoost Top 10 Features:")
    print(xgb_importance.to_string(index=False))

    return best_model_name, comparison


def save_models(logistic_metrics, rf_metrics, xgb_metrics):
    """
    Save trained models to disk

    Args:
        All model metrics dicts
    """
    import os

    os.makedirs(MODELS_PATH, exist_ok=True)

    # Save all models
    with open(f"{MODELS_PATH}logistic_regression.pkl", "wb") as f:
        pickle.dump(logistic_metrics["model"], f)

    with open(f"{MODELS_PATH}random_forest.pkl", "wb") as f:
        pickle.dump(rf_metrics["model"], f)

    with open(f"{MODELS_PATH}xgboost.pkl", "wb") as f:
        pickle.dump(xgb_metrics["model"], f)

    print(f"\n✅ All models saved to {MODELS_PATH}")


def main():
    """
    Main pipeline: Load data → Train all models → Compare → Save
    """
    print("=" * 70)
    print("MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load data
    print("\n1️⃣ LOADING DATA")
    print("-" * 70)
    data = load_processed_data()
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    # 2. Train Logistic Regression
    print("\n2️⃣ TRAINING MODELS")
    print("-" * 70)
    logistic_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)

    # 3. Train Random Forest
    rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)

    # 4. Train XGBoost
    xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)

    # 5. Compare models
    print("\n3️⃣ COMPARING MODELS")
    print("-" * 70)
    best_model, comparison = compare_models(
        logistic_metrics,
        rf_metrics,
        xgb_metrics,
        feature_names=X_train.columns.tolist(),
    )

    # 6. Save models
    print("\n4️⃣ SAVING MODELS")
    print("-" * 70)
    save_models(logistic_metrics, rf_metrics, xgb_metrics)

    # 7. Save comparison results
    comparison.to_csv(f"{MODELS_PATH}model_comparison.csv")
    print(f"✅ Comparison saved to {MODELS_PATH}model_comparison.csv")

    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETE!")
    print(f"   Best model: {best_model}")
    print("=" * 70)

    return logistic_metrics, rf_metrics, xgb_metrics, comparison


if __name__ == "__main__":
    logistic_metrics, rf_metrics, xgb_metrics, comparison = main()
