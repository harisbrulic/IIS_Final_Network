import os
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime

app = FastAPI(title="Cyber Threat Detection API")

autoencoder = load_model("models/autoencoder.keras")
scaler = joblib.load("models/scaler.pkl")
xgboost = joblib.load("models/xgboost.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
ip_classifier = joblib.load("models/ip_classifier.pkl")
ip_label_encoder = joblib.load("models/ip_label_encoder.pkl")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

class TrafficInput(BaseModel):
    features: list[float]
    ip: str = "0.0.0.0"
    port: int = 80
    confidence: float = 100.0
    
class ExplainInput(BaseModel):
    features: list[float]
    
stats = {
    "total_predictions": 0,
    "total_attacks": 0,
    "total_normal": 0,
    "attack_types": {},
    "last_prediction": None
}

@app.get("/")
def root():
    return {"message": "Network Attack"}

@app.post("/predict")
def predict(data: TrafficInput):
    
    ip_parts = data.ip.split(".")
    ip_features = [int(p) for p in ip_parts] + [data.port, data.confidence]
    ip_pred = ip_classifier.predict([ip_features])[0]
    ip_threat = ip_label_encoder.inverse_transform([ip_pred])[0]
    
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_reshaped = X_scaled.reshape(1, 1, X_scaled.shape[1])
    X_pred = autoencoder.predict(X_reshaped)
    error = float(np.mean(np.power(X_reshaped - X_pred, 2)))
    is_anomaly = error > threshold
    
    attack_type = "BENIGN"
    if is_anomaly:
        xgb_pred = xgboost.predict([data.features])[0]
        attack_type = label_encoder.inverse_transform([xgb_pred])[0]
    
    return {
        "is_anomaly": bool(is_anomaly),
        "reconstruction_error": error,
        "attack_type": attack_type,
        "ip_threat": ip_threat
    }
    
    stats["total_predictions"] += 1
    if is_anomaly:
        stats["total_attacks"] += 1
        stats["attack_types"][attack_type] = stats["attack_types"].get(attack_type, 0) + 1
    else:
        stats["total_normal"] += 1
    stats["last_prediction"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "is_anomaly": bool(is_anomaly),
        "reconstruction_error": error,
        "attack_type": attack_type,
        "ip_threat": ip_threat
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/explain")
def explain(data: ExplainInput):
    X = np.array(data.features).reshape(1, -1)
    
    feature_names = [
        "Header_Length", "Protocol", "TTL", "Rate",
        "fin_flag", "syn_flag", "rst_flag", "psh_flag",
        "ack_flag", "ece_flag", "cwr_flag", "ack_count",
        "syn_count", "fin_count", "rst_count", "HTTP",
        "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
        "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP",
        "IPv", "LLC", "Tot_sum", "Min", "Max", "AVG",
        "Std", "Tot_size", "IAT", "Number", "Variance"
    ]
    
    explainer = LimeTabularExplainer(
        training_data=np.zeros((100, len(data.features))),
        feature_names=feature_names,  # ← dodaj ovo!
        mode="classification"
    )
    
    explanation = explainer.explain_instance(
        X[0],
        xgboost.predict_proba,
        num_features=10
    )
    
    lime_values = dict(explanation.as_list())
    return {"lime_values": lime_values}

@app.get("/stats")
def get_stats():
    return stats

