
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 55)
print("FINANCIAL FRAUD DETECTION - DATA PREPROCESSING")
print("=" * 55)

# 1. Load dataset
df = pd.read_csv("data/creditcard.csv")

print("\nOriginal Dataset Shape:", df.shape)

# 2. Check duplicate records
duplicate_count = df.duplicated().sum()

print("\nDuplicate Records:", duplicate_count)

# Remove exact duplicate records
df = df.drop_duplicates().reset_index(drop=True)

print("Dataset Shape After Duplicate Removal:", df.shape)

# 3. Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# 4. Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 5. Scale Time and Amount features
scaler = StandardScaler()

scale_columns = ["Time", "Amount"]

X_train = X_train.copy()
X_test = X_test.copy()

X_train[scale_columns] = scaler.fit_transform(
    X_train[scale_columns]
)

X_test[scale_columns] = scaler.transform(
    X_test[scale_columns]
)

# 6. Create processed data directory
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 7. Save processed datasets
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# 8. Save scaler for future predictions
joblib.dump(scaler, "models/scaler.pkl")

# 9. Display class distribution
print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

print("\nScaling Information:")
print("Scaled Columns:", scale_columns)

print("\nPreprocessing completed successfully!")
print("Processed datasets saved in data/processed/")
print("Scaler saved in models/scaler.pkl")

print("=" * 55)
