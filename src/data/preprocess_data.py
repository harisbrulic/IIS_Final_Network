import os
import pandas as pd
import numpy as np
import json

def preprocess_data():
    
    os.makedirs("data/preprocessed", exist_ok=True)
    
    df = pd.read_csv("data/raw/Merged63.csv")
    df = df.dropna(subset=['Label'])
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    normal_df = df[df['Label'] == 'BENIGN']
    normal_df.to_csv("data/preprocessed/normal.csv", index=False)
    
    attacks_df = df[df['Label'] != 'BENIGN']
    attacks_df.to_csv("data/preprocessed/attacks.csv", index=False)
    
    with open("data/raw/alienvault.json", "r") as f:
        threats = json.load(f)
        
    ips = []
    
    for pulse in threats.get("results",[]):
        for indicator in pulse.get("indicators", []):
            if indicator.get("type") == "IPv4":
                ips.append({
                    
                    "ip": indicator.get("indicator"),
                    "description": indicator.get("description"),
                    "pulse": pulse.get("name")
                })
                
    threats_df = pd.DataFrame(ips)
    threats_df.to_csv("data/preprocessed/threats.csv", index = False)

    return True

if __name__ == "__main__":
    sucess = preprocess_data()
    