from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report


def main():
    model_path = Path("models/model.joblib")
    x_test_path = Path("results/X_test.csv")
    y_test_path = Path("results/y_test.csv")
    metrics_path = Path("results/metrics.json")
    report_path = Path("results/classification_report.txt")

    model = joblib.load(model_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    metrics = {
        "accuracy": float(accuracy)
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("Evaluation finished.")
    print(f"Accuracy: {accuracy:.4f}")
    print("Saved metrics to results/metrics.json")
    print("Saved report to results/classification_report.txt")


if __name__ == "__main__":
    main()
