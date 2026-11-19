
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Professional chart settings
sns.set_theme(style="whitegrid", palette="deep")

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Create output directory
os.makedirs("outputs", exist_ok=True)

print("Starting Exploratory Data Analysis...")

# 1. Transaction Class Distribution
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Class")

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.savefig("outputs/transaction_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 2. Transaction Amount Distribution
plt.figure(figsize=(9, 5))

sns.histplot(data=df, x="Amount", bins=50, kde=True)

plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.savefig("outputs/transaction_amount_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 3. Transaction Amount by Class
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Class", y="Amount")

plt.yscale("log")
plt.title("Transaction Amount by Class (Log Scale)")
plt.xlabel("Transaction Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Transaction Amount (Log Scale)")

plt.savefig("outputs/amount_by_class.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 4. Transaction Time Distribution
plt.figure(figsize=(9, 5))

sns.histplot(data=df, x="Time", bins=50, kde=True)

plt.title("Distribution of Transaction Time")
plt.xlabel("Time (Seconds from Dataset Start)")
plt.ylabel("Number of Transactions")

plt.savefig("outputs/transaction_time_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(16, 12))

correlation = df.corr(numeric_only=True)

sns.heatmap(correlation, cmap="coolwarm",
            center=0, linewidths=0.1)

plt.title("Transaction Feature Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png",
            dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nEDA completed successfully!")
print("All five graphs saved in the outputs folder.")
