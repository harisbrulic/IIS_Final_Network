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
    
    with open("data/raw/threatfox.json", "r") as f:
        data = json.load(f)
        
    threats_path = "data/preprocessed/threats.csv"
    
    if os.path.exists(threats_path):
        existing_df = pd.read_csv(threats_path)
    else:
        existing_df = pd.DataFrame(columns=["ip", "port", "threat_type", "malware", "confidence", "first_seen"])
        
    iocs = []
    for ioc in data.get("data", []):
    
        ioc_type = ioc.get("ioc_type", "")
    
        if ioc_type == "ip:port":
            ioc_value = ioc.get("ioc", "")
        
            if ":" in ioc_value:
                ip, port = ioc_value.rsplit(":", 1)
            else:
                ip = ioc_value
                port = 0
            
            iocs.append({
                "ip": ip,
                "port": port,
                "threat_type": ioc.get("threat_type"),
                "malware": ioc.get("malware_printable"),
                "confidence": ioc.get("confidence_level"),
                "first_seen": ioc.get("first_seen")
        })
                
    new_df = pd.DataFrame(iocs)
    
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["ip", "port", "malware"])
    combined_df = combined_df.dropna(subset=["malware"])
    
    combined_df.to_csv(threats_path, index=False)

    return True

if __name__ == "__main__":
    sucess = preprocess_data()
    