
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.predict import predict_transaction

# Load actual data containing Class labels
actual_data = pd.read_csv(
    "data/test_transactions_with_labels.csv"
)

# Separate input features and actual answers
X = actual_data.drop(columns=["Class"])
y_actual = actual_data["Class"].astype(int)

# Generate model predictions
predictions = []

print("Evaluating fraud detection model...")

for index, row in X.iterrows():

    result = predict_transaction(row.to_dict())

    predictions.append(
        1 if result["prediction"] == "Potential Fraud" else 0
    )

    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} transactions")

y_predicted = predictions

# Create comparison report
comparison = pd.DataFrame({
    "Transaction_ID": range(1, len(y_actual) + 1),
    "Actual_Class": y_actual.values,
    "Predicted_Class": y_predicted
})

comparison["Result"] = comparison.apply(
    lambda row:
        "Correct Detection"
        if row["Actual_Class"] == row["Predicted_Class"]
        else "Incorrect Prediction",
    axis=1
)

# Save comparison results
comparison.to_csv(
    "outputs/fraud_prediction_comparison.csv",
    index=False
)

# Calculate evaluation metrics
print("\n========== MODEL EVALUATION ==========")

print("Total Transactions:", len(y_actual))
print("Actual Fraud:", int(y_actual.sum()))
print("Actual Normal:", int((y_actual == 0).sum()))

print("Accuracy:", round(accuracy_score(y_actual, y_predicted), 4))
print("Precision:", round(precision_score(
    y_actual, y_predicted, zero_division=0
), 4))
print("Recall:", round(recall_score(
    y_actual, y_predicted, zero_division=0
), 4))
print("F1 Score:", round(f1_score(
    y_actual, y_predicted, zero_division=0
), 4))

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_actual, y_predicted, labels=[0, 1]
).ravel()

print("\n========== CONFUSION MATRIX ==========")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_actual,
    y_predicted,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

print("Comparison saved to outputs/fraud_prediction_comparison.csv")
