
import pandas as pd

# Load original dataset
df = pd.read_csv("data/creditcard.csv")

# Select 1,000 transactions, including known fraud examples
normal = df[df["Class"] == 0].head(900)
fraud = df[df["Class"] == 1].head(100)

# Combine and shuffle
test_data = pd.concat([normal, fraud])
test_data = test_data.sample(frac=1, random_state=42)

# Keep Class separately as the actual answer
test_data.to_csv(
    "data/test_transactions_with_labels.csv",
    index=False
)

# Create upload file without actual answers
upload_data = test_data.drop(columns=["Class"])
upload_data.to_csv(
    "data/test_transactions.csv",
    index=False
)

print("Test dataset created successfully!")
print("Total transactions:", len(test_data))
print("Actual fraud transactions:", int(test_data["Class"].sum()))
print("Actual normal transactions:", int((test_data["Class"] == 0).sum()))
