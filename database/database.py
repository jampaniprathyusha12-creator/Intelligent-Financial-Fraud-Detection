
import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "fraud_detection.db"


def get_connection():
    """Create and return a database connection."""
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    """Create the transaction table if it does not exist."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_time TEXT,
            transaction_amount REAL,
            fraud_probability REAL,
            prediction TEXT,
            risk_level TEXT,
            recommendation TEXT,
            alert TEXT,
            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()

    print("Database initialized successfully!")


def save_transaction(
    transaction_time,
    transaction_amount,
    fraud_probability,
    prediction,
    risk_level,
    recommendation,
    alert
):
    """Save a transaction and its prediction."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO transactions (
            transaction_time,
            transaction_amount,
            fraud_probability,
            prediction,
            risk_level,
            recommendation,
            alert,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(transaction_time),
        float(transaction_amount),
        float(fraud_probability),
        prediction,
        risk_level,
        recommendation,
        alert,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()

    print("Transaction saved successfully!")


def get_all_transactions():
    """Retrieve all stored transactions."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT * FROM transactions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    connection.close()

    return records
