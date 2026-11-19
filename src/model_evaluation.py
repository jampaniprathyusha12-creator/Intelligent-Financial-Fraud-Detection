
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "outputs" / "fraud_prediction_comparison.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "model_evaluation.json"


def generate_evaluation():

    df = pd.read_csv(DATA_PATH)

    actual = df["Actual_Class"]
    predicted = df["Predicted_Class"]

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": round(accuracy_score(actual, predicted) * 100, 2),
        "precision": round(precision_score(actual, predicted, zero_division=0) * 100, 2),
        "recall": round(recall_score(actual, predicted, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(actual, predicted, zero_division=0) * 100, 2),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "total_transactions": int(len(df)),
        "actual_fraud": int(actual.sum()),
        "actual_normal": int((actual == 0).sum()),
        "classification_report": classification_report(
            actual,
            predicted,
            target_names=["Normal", "Fraud"],
            output_dict=True,
            zero_division=0
        )
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model evaluation generated successfully!")
    print(f"Accuracy: {metrics['accuracy']}%")
    print(f"Precision: {metrics['precision']}%")
    print(f"Recall: {metrics['recall']}%")
    print(f"F1 Score: {metrics['f1_score']}%")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_evaluation()
