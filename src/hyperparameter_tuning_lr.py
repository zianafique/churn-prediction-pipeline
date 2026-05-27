"""
Hyperparameter Tuning Module
Optimizes Logistic Regression using GridSearchCV
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pickle
import warnings

warnings.filterwarnings("ignore")
import time

# Configuration
RANDOM_STATE = 42
DATA_PATH = "C:/Users/afiqu/Desktop/Learning And Evaulating Python/churn-prediction-pipeline/data/processed/"
MODELS_PATH = "C:/Users/afiqu/Desktop/Learning And Evaulating Python/churn-prediction-pipeline/models/"

print("✅ Hyperparameter tuning module imported")


def load_data():
    """Load training and validation data"""
    X_train = pd.read_csv(f"{DATA_PATH}X_train.csv")
    X_val = pd.read_csv(f"{DATA_PATH}X_val.csv")
    y_train = pd.read_csv(f"{DATA_PATH}y_train.csv").squeeze()
    y_val = pd.read_csv(f"{DATA_PATH}y_val.csv").squeeze()

    print(f"✅ Data loaded:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")

    return X_train, X_val, y_train, y_val


def define_param_grid():
    """Define Logistic Regression hyperparameter search space"""
    param_grid = {
        "C": [
            0.001,
            0.01,
            0.1,
            1,
            10,
            100,
        ],  # Regularization strength (lower = more regularization)
        "penalty": ["l2"],  # L2 regularization
        "solver": ["lbfgs", "liblinear"],  # Different solvers
        "max_iter": [200, 500, 1000],  # Max iterations to converge
    }

    print(f"✅ Parameter grid defined")
    print(f"   Total combinations: {6 * 1 * 2 * 3} = 36")
    print(f"   With 5-fold CV: 180 models to train")
    print(f"   This will be MUCH faster than XGBoost!")

    return param_grid


def load_baseline_model():
    """Load the baseline Logistic Regression model from Step 5"""
    with open(f"{MODELS_PATH}logistic_regression.pkl", "rb") as f:
        baseline_model = pickle.load(f)

    print(f"✅ Baseline Logistic Regression loaded")
    return baseline_model


def tune_logistic_regression(X_train, y_train, X_val, y_val, param_grid):
    """Perform hyperparameter tuning with GridSearchCV"""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING - Logistic Regression")
    print("=" * 70)

    # Create base Logistic Regression model
    base_lr = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1)

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring="roc_auc",  # Optimize for ROC AUC
        n_jobs=-1,  # Use all cores
        verbose=1,
    )

    print(f"\n🔄 Starting GridSearchCV (this will take 5-10 minutes)...")
    start_time = time.time()

    # Fit grid search
    grid_search.fit(X_train, y_train)

    elapsed_time = time.time() - start_time
    print(f"\n✅ GridSearchCV complete! Time: {elapsed_time / 60:.1f} minutes")

    # Best parameters and score
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"\n✅ Best Parameters Found:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"\n✅ Best CV Score (5-fold): {best_cv_score:.4f}")

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, pos_label="Yes")
    val_recall = recall_score(y_val, y_val_pred, pos_label="Yes")
    val_f1 = f1_score(y_val, y_val_pred, pos_label="Yes")

    print(f"\n✅ Validation Set Performance:")
    print(f"   ROC AUC:   {val_auc:.4f}")
    print(f"   Accuracy:  {val_accuracy:.4f}")
    print(f"   Precision: {val_precision:.4f}")
    print(f"   Recall:    {val_recall:.4f}")
    print(f"   F1 Score:  {val_f1:.4f}")

    results = {
        "best_model": best_model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "val_auc": val_auc,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "grid_search": grid_search,
    }

    return results


def compare_baseline_vs_tuned(baseline_model, tuned_results, X_val, y_val):
    """Compare baseline model with tuned model"""
    print("\n" + "=" * 70)
    print("BASELINE vs TUNED COMPARISON")
    print("=" * 70)

    # Baseline predictions
    baseline_pred_proba = baseline_model.predict_proba(X_val)[:, 1]
    baseline_auc = roc_auc_score(y_val, baseline_pred_proba)
    baseline_accuracy = accuracy_score(y_val, baseline_model.predict(X_val))

    # Tuned predictions
    tuned_auc = tuned_results["val_auc"]
    tuned_accuracy = tuned_results["val_accuracy"]

    # Create comparison table
    comparison = pd.DataFrame({
        "Metric": ["ROC AUC", "Accuracy"],
        "Baseline": [baseline_auc, baseline_accuracy],
        "Tuned": [tuned_auc, tuned_accuracy],
        "Improvement": [tuned_auc - baseline_auc, tuned_accuracy - baseline_accuracy],
    })

    print("\n" + comparison.to_string(index=False))

    improvement_auc = tuned_auc - baseline_auc
    print(f"\n✅ ROC AUC Improvement: {improvement_auc:+.4f}")

    if tuned_auc > baseline_auc:
        print(f"   ✅ Tuning was successful!")
    else:
        print(f"   ⚠️ Tuning didn't improve. Baseline was already good.")

    return comparison


def save_tuned_model(tuned_results, best_params, comparison):
    """Save tuned model and results"""
    import os

    os.makedirs(MODELS_PATH, exist_ok=True)

    # Save tuned model
    with open(f"{MODELS_PATH}logistic_regression_tuned.pkl", "wb") as f:
        pickle.dump(tuned_results["best_model"], f)

    # Save best parameters
    with open(f"{MODELS_PATH}best_params_lr.pkl", "wb") as f:
        pickle.dump(best_params, f)

    # Save comparison
    comparison.to_csv(f"{MODELS_PATH}tuning_comparison_lr.csv", index=False)

    print(
        f"\n✅ Tuned Logistic Regression saved to {MODELS_PATH}logistic_regression_tuned.pkl"
    )
    print(f"✅ Best parameters saved")
    print(f"✅ Comparison saved to {MODELS_PATH}tuning_comparison_lr.csv")


def print_best_parameters(best_params):
    """Print best parameters in readable format"""
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)

    print("\nOptimized Logistic Regression Configuration:")
    print(f"""
LogisticRegression(
    C={best_params.get("C", 1)},
    penalty='{best_params.get("penalty", "l2")}',
    solver='{best_params.get("solver", "lbfgs")}',
    max_iter={best_params.get("max_iter", 100)},
    random_state=42,
    n_jobs=-1
)
    """)


def main():
    """Main tuning pipeline"""
    print("=" * 70)
    print("LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("=" * 70)

    # 1. Load data
    print("\n1️⃣ LOADING DATA")
    print("-" * 70)
    X_train, X_val, y_train, y_val = load_data()

    # 2. Define parameter grid
    print("\n2️⃣ DEFINING PARAMETER GRID")
    print("-" * 70)
    param_grid = define_param_grid()

    # 3. Perform tuning
    print("\n3️⃣ PERFORMING HYPERPARAMETER TUNING")
    print("-" * 70)
    tuned_results = tune_logistic_regression(X_train, y_train, X_val, y_val, param_grid)

    # 4. Load baseline and compare
    print("\n4️⃣ COMPARING WITH BASELINE")
    print("-" * 70)
    baseline_model = load_baseline_model()
    comparison = compare_baseline_vs_tuned(baseline_model, tuned_results, X_val, y_val)

    # 5. Save tuned model
    print("\n5️⃣ SAVING TUNED MODEL")
    print("-" * 70)
    save_tuned_model(tuned_results, tuned_results["best_params"], comparison)

    # 6. Print best parameters
    print("\n6️⃣ BEST PARAMETERS")
    print("-" * 70)
    print_best_parameters(tuned_results["best_params"])

    print("\n" + "=" * 70)
    print("✅ HYPERPARAMETER TUNING COMPLETE!")
    print("=" * 70)

    return tuned_results


if __name__ == "__main__":
    tuned_results = main()
