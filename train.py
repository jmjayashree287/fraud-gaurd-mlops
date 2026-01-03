import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import joblib

# 1. Start MLflow Context
mlflow.set_experiment("Fraud_Detection_Production")

with mlflow.start_run():
    # Simulate Banking Data (Highly Imbalanced: 99% Legit, 1% Fraud)
    # Features: V1-V20 (Anonymized), Amount, Transaction_Time
    X, y = make_classification(n_samples=10000, n_features=22, n_informative=20, 
                               n_redundant=0, weights=[0.90, 0.10], flip_y=0)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Train Model (Random Forest is robust for tabular financial data)
    params = {"n_estimators": 50, "max_depth": 10, "class_weight": "balanced"}
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)

    # Predict
    predictions = clf.predict(X_test)

    # Log SRE/Business Metrics
    # In banking, Recall is critical (don't miss fraud), but Precision matters (don't block good users)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    mlflow.log_params(params)
    mlflow.log_metrics({"precision": precision, "recall": recall})
    
    # Save Artifacts
    joblib.dump(clf, "fraud_model.joblib")
    mlflow.sklearn.log_model(clf, "fraud_model")
    
    print(f"Model Trained - Precision: {precision:.2f}, Recall: {recall:.2f}")