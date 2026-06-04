import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def alienvault_fetch():
    
    os.makedirs("data/raw", exist_ok=True)
    api_key = os.getenv("ALIENVAULT_API_KEY")
    
    url = "https://otx.alienvault.com/api/v1/pulses/subscribed"
    headers = {"X-OTX-API-KEY": api_key}
    
    response = requests.get(url, headers=headers)
    
    data = response.json()
    
    file_path = "data/raw/alienvault.json"
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
        
    return True

if __name__ == "__main__":
    sucess = alienvault_fetch()
    
    if sucess:
        print("Data is downloaded")
    
    else:
        print("Data is not downloaded")

    
    