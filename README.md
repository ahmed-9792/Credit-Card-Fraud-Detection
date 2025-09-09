💳 Credit Card Fraud Detection System:

A comprehensive machine learning pipeline for detecting fraudulent credit card transactions using various classification algorithms. The system processes transaction data and outputs predictions in a predictions.csv file.

📋 Project Overview
This project provides an end-to-end solution for credit card fraud detection, featuring:

Data preprocessing and feature engineering

Multiple machine learning models

Model evaluation and selection

Streamlit web application for real-time predictions

CSV output generation with fraud predictions

🏗️ Project Structure

fraud_detection_project/
├── src/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── model_training.py        # Model training functions
│   ├── model_evaluation.py      # Model evaluation utilities
│   └── __init__.py
├── models/                      # Saved models and scalers
├── data/
│   └── raw/                     # Raw dataset directory
├── tests/
│   ├── test_data_preprocessing.py
│   └── test_model_training.py
├── fraud_app.py                 # Streamlit web application
├── run_pipeline.py             # Main training pipeline
├── setup.py                    # Package configuration
├── predictions.csv             # OUTPUT: Predictions file (generated)
└── requirements.txt

🚀 Installation
Clone the repository:
git clone <repository-url>
cd fraud_detection_project

Install dependencies:
pip install -r requirements.txt

Install the package:
pip install -e .

📊 Dataset
The system expects a CSV file with the following structure:

Time: Time since first transaction

V1-V28: Anonymized principal components

Amount: Transaction amount

Class: Target variable (0: legitimate, 1: fraudulent)

Place your dataset at data/raw/creditcard.csv

🛠️ Usage
Training the Models and Generating Predictions

Run the complete training pipeline:
python run_pipeline.py

This will:

Load and preprocess the data

Train multiple classification models

Evaluate and select the best model

Save all models and preprocessing artifacts

Generate predictions.csv with fraud predictions

Output File: predictions.csv
The pipeline generates a predictions.csv file with the following structure:

Time	V1	V2	...	V28	Amount	Fraud_Prediction
0.496	-0.138	0.647	...	-0.600	-0.291	1
-0.601	1.852	-0.013	...	0.331	0.975	1
...	...	...	...	...	...	...
Where:

Fraud_Prediction: 1 = Fraudulent transaction, 0 = Legitimate transaction

All original features are preserved for reference

Using the Web Application

Launch the Streamlit dashboard:
streamlit run fraud_app.py
The application provides:

File upload functionality for CSV transactions

Real-time fraud predictions

Visual results with probability scores

Downloadable predictions in CSV format

Individual Components
You can also use individual modules:
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_and_save_models, evaluate_models

# Load and preprocess data
df = load_data("data/raw/creditcard.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train models
models = train_and_save_models(X_train, y_train)

# Evaluate performance
results = evaluate_models(models, X_test, y_test)

# Generate predictions (example)
best_model = models["xgboost"]  # or whichever performed best
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

# Create predictions DataFrame
output_df = X_test.copy()
output_df["Fraud_Prediction"] = predictions
output_df["Fraud_Probability"] = probabilities
output_df.to_csv("predictions.csv", index=False)

🤖 Available Models
The system trains and evaluates four different algorithms:

Logistic Regression - Baseline model

Random Forest - Ensemble method

Support Vector Machine - Kernel-based classifier

XGBoost - Gradient boosting implementation

⚙️ Configuration
Preprocessing Settings
Data standardization using StandardScaler

Handling class imbalance with SMOTE

Train-test split with stratification

Random state fixed for reproducibility

Model Parameters
Random Forest: 100 estimators

Logistic Regression: 500 maximum iterations

XGBoost: Default parameters with logloss evaluation

SVM: RBF kernel with probability estimates

📈 Performance Metrics
Models are evaluated using:

Accuracy score

ROC-AUC score

Classification report (precision, recall, f1-score)

🧪 Testing
Run the test suite:
python -m pytest tests/

Tests include:
Data loading functionality

Model training procedures

Input validation

Prediction output format validation

🎯 Results Interpretation
The predictions.csv file contains:

All original features from input data

Fraud_Prediction: Binary classification (0 = legitimate, 1 = fraudulent)

Additional columns may include probability scores depending on configuration

📝 Output Files
The pipeline generates:

Trained model files (.pkl)

Standard scaler object

Feature list (JSON)

predictions.csv - Primary output with fraud predictions

🔮 Future Enhancements
Potential improvements:

Real-time API endpoints

Advanced anomaly detection algorithms

Model explainability (SHAP/LIME)

Database integration
