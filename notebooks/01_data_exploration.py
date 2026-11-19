
import pandas as pd
import os

# Dataset location
file_path = "data/creditcard.csv"

# Check whether dataset exists
if os.path.exists(file_path):

    # Load dataset
    df = pd.read_csv(file_path)

    print("\n========== DATASET LOADED SUCCESSFULLY ==========")

    # Display first five transactions
    print("\nFirst 5 Transactions:")
    print(df.head())

    # Display dataset shape
    print("\nDataset Shape:")
    print(df.shape)

    # Display column names
    print("\nColumn Names:")
    print(df.columns.tolist())

    # Display dataset information
    print("\nDataset Information:")
    print(df.info())

    # Check missing values
    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    # Count normal and fraudulent transactions
    print("\nTransaction Distribution:")
    print(df["Class"].value_counts())

    # Fraud percentage
    fraud_percentage = (df["Class"].mean()) * 100

    print("\nFraud Percentage:")
    print(round(fraud_percentage, 4), "%")

else:
    print("Dataset file not found. Check the data folder.")
