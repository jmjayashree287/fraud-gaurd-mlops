from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import joblib
import time
import numpy as np

app = FastAPI(title="FraudGuard Inference Engine")
model = joblib.load("fraud_model.joblib")

# --- DATA SCHEMA ---
class Transaction(BaseModel):
    amount: float
    features: list  # The 21 other features

# --- SRE METRICS (Banking Specific) ---

# 1. Latency: Critical. If > 200ms, the payment gateway times out.
TXN_LATENCY = Histogram(
    'txn_processing_seconds', 
    'Time taken to score a transaction',
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0] # Finely grained buckets for P99 tuning
)

# 2. Fraud Counter: Tracks volume of blocked transactions
FRAUD_ALERTS = Counter(
    'fraud_detected_total', 
    'Total blocked transactions',
    ['risk_level'] # Labels: High, Medium, Low
)

# 3. Financial Impact: Tracks total money processed vs blocked
MONEY_PROCESSED = Counter(
    'money_processed_total', 
    'Total transaction volume processed', 
    ['status'] # status: Approved / Declined
)

@app.post("/score_transaction")
def score(txn: Transaction):
    start_time = time.time()
    
    try:
        # Preprocess
        data = np.array(txn.features + [txn.amount]).reshape(1, -1)
        
        # Inference
        is_fraud = model.predict(data)[0]
        
        # Record Metrics
        status = "Declined" if is_fraud else "Approved"
        MONEY_PROCESSED.labels(status=status).inc(txn.amount)
        
        if is_fraud:
            FRAUD_ALERTS.labels(risk_level="High").inc()

        # Latency Observation
        TXN_LATENCY.observe(time.time() - start_time)
        
        return {"transaction_id": "tx_123", "status": status, "fraud_score": int(is_fraud)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST) 