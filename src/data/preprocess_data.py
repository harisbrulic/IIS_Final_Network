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
        
    threats_path = "data/preprocessed/threats.csv"
    
    if os.path.exists(threats_path):
        existing_df = pd.read_csv(threats_path)
    else:
        existing_df = pd.DataFrame(columns=["ip","pulse"])
        
    ips = []
    
    for pulse in threats.get("results",[]):
        for indicator in pulse.get("indicators", []):
            if indicator.get("type") == "IPv4":
                ips.append({
                    
                    "ip": indicator.get("indicator"),
                    "pulse": pulse.get("name"),
                    "created": indicator.get("created")
                })
                
    new_df = pd.DataFrame(ips)
    
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["ip", "pulse"])
    
    combined_df.to_csv(threats_path, index=False)

    return True

if __name__ == "__main__":
    sucess = preprocess_data()
    