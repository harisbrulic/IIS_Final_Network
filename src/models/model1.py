import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

def model1():
    
    os.makedirs("models", exist_ok=True)
    
    df = pd.read_csv("data/preprocessed/normal.csv")
    df = df.select_dtypes(include=[np.number])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    mlflow.set_experiment("autoencoder")
    
    with mlflow.start_run():
        
        timesteps = 1
        n_features = df.shape[1]
        encoding_dim = 32
        epochs = 50
        batch_size = 256
        
        mlflow.log_param("timesteps", timesteps)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("encoding_dim", encoding_dim)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        inputs = Input(shape=(timesteps, n_features))
        encoded = LSTM(encoding_dim, activation='relu')(inputs)
        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(n_features, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(n_features))(decoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.0001
        )
        
        history = autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )
        
        mlflow.log_metric("final_loss", history.history['loss'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("epochs_trained", len(history.history['loss']))
        mlflow.log_metric("best_val_loss", min(history.history['val_loss']))
        
        X_pred = autoencoder.predict(X)
        reconstruction_errors = np.mean(np.power(X - X_pred, 2), axis=(1,2))
        threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        
        mlflow.log_metric("mean_reconstruction_error", float(np.mean(reconstruction_errors)))
        mlflow.log_metric("threshold", float(threshold))
        
        attacks_df = pd.read_csv("data/preprocessed/attacks.csv")
        attacks_df = attacks_df.select_dtypes(include=[np.number])
        X_attacks = scaler.transform(attacks_df)
        X_attacks = X_attacks.reshape(X_attacks.shape[0], 1, X_attacks.shape[1])
        
        X_attacks_pred = autoencoder.predict(X_attacks)
        attack_errors = np.mean(np.power(X_attacks - X_attacks_pred, 2), axis=(1,2))
        
        detected = np.sum(attack_errors > threshold)
        detection_rate = float(detected / len(attack_errors))
        
        mlflow.log_metric("detection_rate", detection_rate)
        print(f"Detection rate: {detection_rate:.2%}")
        
        autoencoder.save("models/autoencoder.keras")
        joblib.dump(scaler, "models/scaler.pkl")
        
        with open("models/threshold.json", "w") as f:
            json.dump({"threshold": float(threshold)}, f)
        
        mlflow.keras.log_model(autoencoder, "autoencoder")
        
        print(f"Loss: {history.history['loss'][-1]:.4f}")
        print("Done")
    
    return True

if __name__ == "__main__":
    model1()