import os
import json
import urllib3
from dotenv import load_dotenv

load_dotenv()

def threatfox_fetch():
    
    os.makedirs("data/raw", exist_ok=True)
    
    api_key = os.getenv("THREATFOX_API_KEY")
    
    headers = {
        "Auth-Key": api_key
    }
    
    pool = urllib3.HTTPSConnectionPool(
        'threatfox-api.abuse.ch',
        port=443,
        maxsize=50,
        headers=headers
    )
    
    payload = json.dumps({
        "query": "get_iocs",
        "days": 1
    })
    
    response = pool.request("POST", "/api/v1/", body=payload)
    
    data = json.loads(response.data.decode("utf-8"))
    print(f"Status: {data.get('query_status')}")
    print(f"IoC-ova: {len(data.get('data', []))}")
    
    with open("data/raw/threatfox.json", "w") as f:
        json.dump(data, f, indent=2)
    
    return True

if __name__ == "__main__":
    success = threatfox_fetch()
    if success:
        print("Done")
    else:
        print("Failed")