import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

def ip_to_features(ip):
    try:
        parts = ip.split(".")
        return [int(p) for p in parts]
    except:
        return [0, 0, 0, 0]

def train_ip_classifier():
    
    os.makedirs("models", exist_ok=True)
    
    df = pd.read_csv("data/preprocessed/threats.csv")

    ip_features = df['ip'].apply(ip_to_features)
    X = pd.DataFrame(
        ip_features.tolist(),
        columns=['oct1', 'oct2', 'oct3', 'oct4']
    )
    X['port'] = df['port']
    X['confidence'] = df['confidence']
    
    y = df['threat_type']
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    mlflow.set_experiment("ip_classifier")
    
    with mlflow.start_run():
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1: {f1:.4f}")
        
        joblib.dump(model, "models/ip_classifier.pkl")
        joblib.dump(le, "models/ip_label_encoder.pkl")
        
        mlflow.sklearn.log_model(model, "ip_classifier")
        
        print("Done")
    
    return True

if __name__ == "__main__":
    train_ip_classifier()