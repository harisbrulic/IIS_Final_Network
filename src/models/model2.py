import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

def model2():
    
    os.makedirs("models", exist_ok=True)
    
    normal = pd.read_csv("data/preprocessed/normal.csv")
    attacks = pd.read_csv("data/preprocessed/attacks.csv")
    
    df = pd.concat([normal, attacks], ignore_index=True)
    
    X = df.select_dtypes(include=[np.number])
    y = df['Label']
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    mlflow.set_experiment("xgboost")
    
    with mlflow.start_run():
        
        n_estimators = 100
        max_depth = 6
        learning_rate = 0.1
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        joblib.dump(model, "models/xgboost.pkl")
        joblib.dump(le, "models/label_encoder.pkl")
        
        mlflow.sklearn.log_model(model, "xgboost")
        
        print("Done")
    
    return True

if __name__ == "__main__":
    model2()