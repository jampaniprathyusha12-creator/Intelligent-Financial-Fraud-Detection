
from database.database import get_all_transactions
from src.report_generator import generate_fraud_report

transactions = get_all_transactions()

if transactions:
    report = generate_fraud_report(transactions)
    print("PDF report generated successfully!")
    print("Report location:", report)
else:
    print("No transaction records found. Run fraud detection first.")
