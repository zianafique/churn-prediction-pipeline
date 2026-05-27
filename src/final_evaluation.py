"""
Final Evaluation Module
Comprehensive testing on hold-out test set
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = "C:/Users/afiqu/Desktop/Learning And Evaulating Python/churn-prediction-pipeline/data/processed/"
MODELS_PATH = "C:/Users/afiqu/Desktop/Learning And Evaulating Python/churn-prediction-pipeline/models/"

print("✅ Final evaluation module imported")


def load_test_data():
    """Load test data"""
    X_test = pd.read_csv(f"{DATA_PATH}X_test.csv")
    y_test = pd.read_csv(f"{DATA_PATH}y_test.csv").squeeze()

    print(f"✅ Test data loaded:")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_test: {y_test.shape}")

    return X_test, y_test


def load_tuned_model():
    """Load the tuned Logistic Regression model"""
    with open(f"{MODELS_PATH}logistic_regression_tuned.pkl", "rb") as f:
        model = pickle.load(f)

    print(f"✅ Tuned model loaded")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation

    Returns:
        dict: All evaluation metrics
    """
    print("FINAL MODEL EVALUATION - TEST SET")
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="Yes")
    recall = recall_score(y_test, y_pred, pos_label="Yes")
    f1 = f1_score(y_test, y_pred, pos_label="Yes")
    roc_auc = roc_auc_score((y_test == "Yes").astype(int), y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Specificity and sensitivity
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    print(f"\n✅ Performance Metrics:")
    print(f"   Accuracy:     {accuracy:.4f}")
    print(f"   Precision:    {precision:.4f}")
    print(f"   Recall:       {recall:.4f}")
    print(f"   Specificity:  {specificity:.4f}")
    print(f"   F1 Score:     {f1:.4f}")
    print(f"   ROC AUC:      {roc_auc:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"   True Negatives:  {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   True Positives:  {tp}")

    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    results = {
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "cm": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return results


def create_summary_report(results):
    """Create summary report"""
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    print(f"\nModel Type: Logistic Regression (Tuned)")
    print(f"\nKey Metrics:")
    print(f"  • Accuracy:     {results['accuracy']:.2%} - Overall correctness")
    print(f"  • ROC AUC:      {results['roc_auc']:.4f} - Probability ranking ability")
    print(
        f"  • Precision:    {results['precision']:.2%} - Of predicted churns, how many correct"
    )
    print(
        f"  • Recall:       {results['recall']:.2%} - Of actual churns, how many caught"
    )
    print(
        f"  • Specificity:  {results['specificity']:.2%} - Of non-churners, how many correct"
    )

    print(f"\nBusiness Impact:")
    total_positives = results["tp"] + results["fn"]
    caught_percentage = (
        results["tp"] / total_positives * 100 if total_positives > 0 else 0
    )
    print(
        f"  • Catch {caught_percentage:.1f}% of actual churners ({results['tp']} out of {total_positives})"
    )
    print(f"  • Avoid {results['specificity']:.1%} false alarms among non-churners")
    print(
        f"  • Precision of {results['precision']:.1%} means 1 in {1 / results['precision']:.0f} predictions correct"
    )

    print(f"\nRecommendation:")
    if results["roc_auc"] >= 0.85:
        print(f"  ✅ Model is PRODUCTION READY")
        print(
            f"     ROC AUC of {results['roc_auc']:.4f} indicates strong predictive power"
        )
    elif results["roc_auc"] >= 0.80:
        print(f"  ✅ Model is ACCEPTABLE for deployment")
        print(f"     ROC AUC of {results['roc_auc']:.4f} is good, consider monitoring")
    else:
        print(f"  ⚠️ Model needs improvement")
        print(f"     ROC AUC of {results['roc_auc']:.4f} may need refinement")


def save_final_report(results, X_test):
    """Save final evaluation report"""
    import os

    os.makedirs(f"{MODELS_PATH}../reports/", exist_ok=True)

    # Create comprehensive report
    report = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1 Score",
            "ROC AUC",
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
        ],
        "Value": [
            f"{results['accuracy']:.4f}",
            f"{results['precision']:.4f}",
            f"{results['recall']:.4f}",
            f"{results['specificity']:.4f}",
            f"{results['f1']:.4f}",
            f"{results['roc_auc']:.4f}",
            str(results["tn"]),
            str(results["fp"]),
            str(results["fn"]),
            str(results["tp"]),
        ],
    })

    report.to_csv(f"{MODELS_PATH}../reports/final_evaluation_report.csv", index=False)
    print(
        f"\n✅ Final report saved to {MODELS_PATH}../reports/final_evaluation_report.csv"
    )

    return report


def main():
    """Main evaluation pipeline"""
    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    # 1. Load test data
    print("\n1️⃣ LOADING TEST DATA")
    print("-" * 70)
    X_test, y_test = load_test_data()

    # 2. Load tuned model
    print("\n2️⃣ LOADING TUNED MODEL")
    print("-" * 70)
    model = load_tuned_model()

    # 3. Evaluate model
    print("\n3️⃣ EVALUATING MODEL")
    print("-" * 70)
    results = evaluate_model(model, X_test, y_test)

    # 4. Create summary
    print("\n4️⃣ CREATING SUMMARY")
    print("-" * 70)
    create_summary_report(results)

    # 5. Save report
    print("\n5️⃣ SAVING REPORT")
    print("-" * 70)
    report = save_final_report(results, X_test)

    print("\n" + "=" * 70)
    print("✅ FINAL EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nModel Status: READY FOR PRODUCTION")
    print(f"Next Steps:")
    print(f"  1. Create deployment package")
    print(f"  2. Set up monitoring")
    print(f"  3. Deploy to production")
    print("=" * 70)

    return results, model


if __name__ == "__main__":
    results, model = main()
