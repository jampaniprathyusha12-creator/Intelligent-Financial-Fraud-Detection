
import pandas as pd
import numpy as np
import joblib
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("INTELLIGENT FINANCIAL FRAUD DETECTION")
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Calculate class imbalance ratio
fraud_count = int(y_train.sum())
normal_count = len(y_train) - fraud_count

scale_pos_weight = normal_count / fraud_count

print("\nNormal Transactions:", normal_count)
print("Fraudulent Transactions:", fraud_count)
print("Class Imbalance Ratio:", round(scale_pos_weight, 2))

# --------------------------------------------------
# 3. INITIALIZE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        max_depth=15,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42
    )
}

results = []
trained_models = {}

# --------------------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

for name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Fraud probability
    probabilities = model.predict_proba(X_test)[:, 1]

    # Default classification threshold
    predictions = (probabilities >= 0.5).astype(int)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    pr_auc = average_precision_score(
        y_test, probabilities
    )
    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "PR AUC": pr_auc,
        "ROC AUC": roc_auc,
        "Training Time (seconds)": training_time
    })

    trained_models[name] = model

    print("\nModel Performance:")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("PR AUC   :", round(pr_auc, 4))
    print("ROC AUC  :", round(roc_auc, 4))
    print("Training Time:", round(training_time, 2), "seconds")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0
    ))

# --------------------------------------------------
# 5. COMPARE MODEL PERFORMANCE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# --------------------------------------------------
# 6. SELECT BEST MODEL USING PR AUC
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "models/best_fraud_model.pkl"
)

print("\nBest Model:", best_model_name)
print("Selection Metric: PR AUC")

# --------------------------------------------------
# 7. SAVE BEST MODEL CONFUSION MATRIX
# --------------------------------------------------

best_probabilities = best_model.predict_proba(X_test)[:, 1]
best_predictions = (best_probabilities >= 0.5).astype(int)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.savefig(
    "outputs/best_model_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("All models trained and compared.")
print("Best model saved in models/best_fraud_model.pkl")
print("Comparison saved in outputs/model_comparison.csv")
print("Confusion matrix saved in outputs/")
print("=" * 60)
