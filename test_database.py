
from database.database import (
    initialize_database,
    save_transaction,
    get_all_transactions
)

initialize_database()

save_transaction(
    transaction_time="10000",
    transaction_amount=2500.00,
    fraud_probability=0.92,
    prediction="Fraud",
    risk_level="High Risk",
    recommendation="Transaction requires manual verification.",
    alert="HIGH PRIORITY: Suspicious transaction detected!"
)

transactions = get_all_transactions()

print("\nStored Transactions:")
for transaction in transactions:
    print(transaction)
