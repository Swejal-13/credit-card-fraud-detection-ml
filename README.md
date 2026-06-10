# Credit Card Fraud Detection using XGBoost

## Overview
This project builds a machine learning model to detect fraudulent credit card transactions using the XGBoost Classifier. The dataset is highly imbalanced, so preprocessing and evaluation techniques are applied to achieve reliable fraud detection performance.

## Features
- Data preprocessing and cleaning
- Feature scaling using StandardScaler
- Handling class imbalance with SMOTE
- Fraud detection using XGBoost Classifier
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score
  - Confusion Matrix

## Dataset
The project uses the Credit Card Fraud Detection dataset containing anonymized transaction features.

Target Variable:
- `Class = 0` → Legitimate Transaction
- `Class = 1` → Fraudulent Transaction

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)
- Matplotlib
- Seaborn
- Google Colab

## Project Workflow

### 1. Data Loading
Load the credit card transaction dataset.

### 2. Data Preprocessing
- Remove unnecessary columns (`Time`)
- Scale the `Amount` feature using StandardScaler

### 3. Train-Test Split
Split the dataset into training and testing sets.

### 4. Handle Class Imbalance
Apply SMOTE (Synthetic Minority Oversampling Technique) to balance fraud and non-fraud samples.

### 5. Model Training
Train an XGBoost Classifier with optimized parameters.

### 6. Model Evaluation
Evaluate performance using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

## Installation

```bash
git clone https://github.com/Swejal-13/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
```

## Run the Project

Open the notebook:

```bash
jupyter notebook CreditCardXGBClassifier.ipynb
```

or run it in Google Colab.

## Model Used

- XGBoost Classifier
  - n_estimators = 100
  - max_depth = 6
  - learning_rate = 0.1

## Results

The XGBoost model achieved excellent performance on the test dataset despite the highly imbalanced nature of credit card fraud transactions.

| Metric | Score |
|----------|----------|
| Accuracy | 99.95% |
| Precision | 92.59% |
| Recall | 76.53% |
| F1-Score | 83.80% |
| ROC-AUC Score | 95.65% |

### Classification Report

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.93      0.77      0.84        98

    accuracy                           1.00     56962
   macro avg       0.96      0.88      0.92     56962
weighted avg       1.00      1.00      1.00     56962
```

### Key Insights

- Achieved **99.95% overall accuracy** on unseen transactions.
- Obtained a **ROC-AUC score of 95.65%**, indicating excellent discrimination between fraudulent and legitimate transactions.
- High **precision (92.59%)** means most flagged fraud transactions are truly fraudulent.
- **Recall of 76.53%** shows the model successfully identifies a large portion of fraudulent transactions.
- The model demonstrates strong performance for real-world fraud detection scenarios where class imbalance is a significant challenge.

