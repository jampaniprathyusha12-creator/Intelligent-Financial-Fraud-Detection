# Sentinel AI – Intelligent Financial Fraud Detection and Prevention System

## Project Overview

Sentinel AI is a machine learning-based financial fraud detection system designed to identify suspicious transactions and support financial risk monitoring.

The system analyzes transaction data, predicts the probability of fraudulent activity, classifies transactions into risk categories, and generates alerts for further investigation.

It provides an interactive Streamlit dashboard for transaction analysis, model evaluation, fraud intelligence, and transaction history management.

## Key Features

- **Exploratory Data Analysis:** Analyzes transaction patterns, distributions, and correlations.
- **Data Preprocessing:** Handles duplicate records, feature separation, train-test splitting, and feature scaling.
- **Machine Learning Models:** Compares Logistic Regression, Random Forest, and XGBoost.
- **Fraud Prediction:** Estimates the probability of suspicious transactions.
- **Risk Classification:** Categorizes transactions into Low, Medium, and High Risk.
- **Fraud Alerts:** Generates alerts for transactions requiring investigation.
- **Interactive Dashboard:** Provides a user-friendly interface using Streamlit.
- **Transaction Database:** Stores analyzed transaction records using SQLite.
- **Model Evaluation:** Displays evaluation metrics and confusion matrices.
- **PDF Report Generation:** Generates downloadable fraud detection reports.
- **CSV Export:** Allows users to download transaction history.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Application | Streamlit |
| Database | SQLite |
| Report Generation | ReportLab |
| Development Environment | VS Code |

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle.

Dataset characteristics:

- 284,807 transactions
- 492 fraudulent transactions
- 284,315 legitimate transactions
- 31 columns
- Contains anonymized transaction features, transaction time, amount, and class labels.

The dataset is highly imbalanced, making precision, recall, F1-score, and Precision-Recall AUC important evaluation measures.

## Project Structure

```text

Intelligent-Financial-Fraud-Detection/
│
├── data/
│   ├── creditcard.csv
│   ├── test_transactions.csv
│   ├── test_transactions_with_labels.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   └── 01_data_exploration.py
│
├── models/
│   ├── scaler.pkl
│   └── best_fraud_model.pkl
│
├── outputs/
│   ├── transaction_distribution.png
│   ├── transaction_amount_distribution.png
│   ├── amount_by_class.png
│   ├── transaction_time_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.csv
│   ├── best_model_confusion_matrix.png
│   ├── fraud_prediction_comparison.csv
│   ├── fraud_confusion_matrix.png
│   ├── model_evaluation.json
│   └── fraud_detection_report.pdf
│
├── database/
│   ├── database.py
│   └── fraud_detection.db
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── predict.py
│   ├── risk_analysis.py
│   ├── evaluation_visualization.py
│   ├── model_evaluation.py
│   └── report_generator.py
│
├── screenshots/
│   ├── 01_project_structure.jpg
│   ├── 02_dataset_preview.jpg
│   ├── 03_preprocessing.jpg
│   ├── 04_model_training.jpg
│   ├── 05_model_comparison.jpg
│   ├── 06_confusion_matrix.jpg
│   ├── 07_evaluation_metrics.jpg
│   ├── 08_database.jpg
│   ├── 09_pdf_report_generation.jpg
│   ├── 10_generated_pdf_report.jpg
│   ├── 11_executive_dashboard.jpg
│   ├── 12_transaction_analysis.jpg
│   ├── 13_fraud_intelligence.jpg
│   ├── 14_model_performance.jpg
│   ├── 15_transaction_history.jpg
│   └── 16_about_project.jpg
│
├── app.py
├── create_test_data.py
├── evaluate_predictions.py
├── test_database.py
├── test_report.py
├── requirements.txt
├── README.md
└── .gitignore
'''

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Intelligent-Financial-Fraud-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Dataset

Download the Credit Card Fraud Detection dataset from Kaggle and place `creditcard.csv` inside the `data/` folder.

## Running the Project

Run the following commands from the project root directory.

### Step 1 – Exploratory Data Analysis

```bash
python src/eda.py
```

### Step 2 – Data Preprocessing

```bash
python src/preprocessing.py
```

### Step 3 – Model Training

```bash
python src/train_model.py
```

### Step 4 – Model Evaluation

```bash
python src/model_evaluation.py
```

### Step 5 – Launch the Dashboard

```bash
streamlit run app.py
```

The application will open in your browser.

## Dashboard Modules

1. **Executive Dashboard** – Displays an overview of fraud detection activity.
2. **Transaction Analysis** – Accepts transaction data and performs fraud prediction.
3. **Fraud Intelligence** – Presents suspicious activity and risk information.
4. **Model Performance** – Displays model evaluation metrics and confusion matrices.
5. **Transaction History** – Shows previously analyzed transactions stored in SQLite and provides CSV/PDF downloads.
6. **About the Project** – Explains the project architecture, workflow, and technology stack.

## Risk Classification

| Risk Level | Interpretation | Recommended Action |
|---|---|---|
| Low Risk | Low predicted fraud probability | Continue normal monitoring |
| Medium Risk | Moderate predicted fraud probability | Additional verification |
| High Risk | High predicted fraud probability | Investigate and generate an alert |

## System Workflow

```text
Transaction Dataset
        |
        v
Data Exploration
        |
        v
Data Preprocessing
        |
        v
Feature Scaling
        |
        v
Machine Learning Models
        |
        v
Fraud Probability Prediction
        |
        v
Risk Classification
        |
        v
Fraud Alert Generation
        |
        v
SQLite Transaction Storage
        |
        v
Streamlit Dashboard
        |
        v
CSV and PDF Reports
```

## Project Outcomes

- Developed a complete machine learning workflow for financial fraud detection.
- Compared multiple classification algorithms.
- Implemented transaction-level fraud probability prediction.
- Developed a risk classification and alert mechanism.
- Integrated SQLite for transaction storage.
- Built an interactive dashboard with downloadable reports.

## Limitations and Future Enhancements

- The dataset contains historical and anonymized transaction information.
- Real-time bank transaction integration is not currently implemented.
- Risk thresholds require further validation and calibration.
- The system is an academic prototype and does not automatically block transactions.

Future enhancements may include real-time payment monitoring, advanced anomaly detection, model explainability, automated model retraining, and integration with financial transaction APIs.

## Conclusion

Sentinel AI demonstrates how machine learning, data analytics, and risk classification can be combined to build an intelligent financial fraud detection system. It provides an integrated platform for analyzing transactions, identifying suspicious patterns, generating alerts, and maintaining transaction records.

---

**Project Category:** Open Innovation

**Project Title:** Intelligent Financial Fraud Detection and Prevention System

**Application Name:** Sentinel AI
:::