import pandas as pd
import joblib
import os
import sys
from pathlib import Path

# Locate the src directory
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Import risk analysis functions
from risk_analysis import calculate_risk, generate_alert

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = "models/best_fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found. Run src/train_model.py first."
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Scaler not found.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_transaction(transaction):

    """
    Predict fraud probability and risk level
    for one transaction.
    """

    # Convert input into DataFrame
    transaction_df = pd.DataFrame([transaction])

    # Scale Time and Amount using saved training scaler
    transaction_df[["Time", "Amount"]] = scaler.transform(
        transaction_df[["Time", "Amount"]]
    )

    # Predict fraud probability
    fraud_probability = model.predict_proba(
        transaction_df
    )[0][1]

    # Classify risk
    risk = calculate_risk(fraud_probability)

    # Generate alert
    alert = generate_alert(
        fraud_probability,
        risk["risk_level"]
    )

    result = {
        "fraud_probability": round(
            fraud_probability * 100, 2
        ),
        "prediction": (
            "Potential Fraud"
            if fraud_probability >= 0.5
            else "Normal Transaction"
        ),
        "risk_level": risk["risk_level"],
        "recommendation": risk["recommendation"],
        "alert": alert
    }

    return result


# --------------------------------------------------
# TEST WITH A REAL DATASET TRANSACTION
# --------------------------------------------------

if __name__ == "__main__":

    test_data = pd.read_csv(
        "data/processed/X_test.csv"
    )

    # Select one transaction for demonstration
    sample_transaction = test_data.iloc[0].to_dict()

    result = predict_transaction(sample_transaction)

    print("\n" + "=" * 50)
    print("FINANCIAL FRAUD DETECTION RESULT")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("=" * 50)
