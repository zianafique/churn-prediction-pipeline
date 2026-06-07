import json
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter


def analyze_metrics(log_file="api_metrics.log", hours=24):
    """Analyze API metrics from log file"""

    predictions = []
    errors = []

    # Read logs
    with open(log_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line.split(" - ", 1)[1])

                if data.get("status") == "success":
                    predictions.append(data)
                else:
                    errors.append(data)
            except:
                continue

    # Analyze
    total_requests = len(predictions) + len(errors)
    error_rate = len(errors) / total_requests if total_requests > 0 else 0

    # Prediction distribution
    predictions_list = [p["prediction"] for p in predictions]
    churn_rate = (
        predictions_list.count("Yes") / len(predictions_list) if predictions else 0
    )

    # Confidence scores
    confidences = [p["probability"] for p in predictions]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_requests": total_requests,
        "successful_requests": len(predictions),
        "errors": len(errors),
        "error_rate": f"{error_rate * 100:.2f}%",
        "predicted_churn_rate": f"{churn_rate * 100:.2f}%",
        "average_confidence": f"{avg_confidence:.4f}",
        "alerts": [],
    }

    # Check for issues
    if error_rate > 0.05:
        report["alerts"].append("⚠️ High error rate!")

    if churn_rate > 0.5:
        report["alerts"].append("⚠️ Unusual churn rate!")

    if avg_confidence < 0.6:
        report["alerts"].append("⚠️ Low prediction confidence!")

    return report


def print_report():
    print("Analyzing API metrics...")
    report = analyze_metrics()
    print("\n" + "=" * 60)
    print("📊 API MONITORING REPORT")
    print("=" * 60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Total Requests: {report['total_requests']}")
    print(f"Successful: {report['successful_requests']}")
    print(f"Errors: {report['errors']}")
    print(f"Error Rate: {report['error_rate']}")
    print(f"Predicted Churn Rate: {report['predicted_churn_rate']}")
    print(f"Average Confidence: {report['average_confidence']}")

    if report["alerts"]:
        print(f"\n🚨 ALERTS:")
        for alert in report["alerts"]:
            print(f"  {alert}")
    else:
        print("\n✅ No alerts - API healthy!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_report()
