
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Load prediction comparison
df = pd.read_csv("outputs/fraud_prediction_comparison.csv")

# Generate confusion matrix
matrix = confusion_matrix(
    df["Actual_Class"],
    df["Predicted_Class"]
)

# Professional visualization
plt.figure(figsize=(8, 6))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title(
    "Financial Fraud Detection - Confusion Matrix",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

Path("outputs").mkdir(exist_ok=True)

plt.savefig(
    "outputs/fraud_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Confusion matrix saved successfully!")
