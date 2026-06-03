import requests
import json
import time

# In Colab, localhost or 0.0.0.0 is often more reliable than 127.0.0.1 for internal requests
url = "http://localhost:8000/predict"

payload = {
    "lag_1": 50000.0,
    "lag_2": 45000.0,
    "lag_3": 60000.0,
    "rolling_mean_7": 52000.0,
    "rolling_mean_30": 50000.0,
    "day_of_week": 2,
    "month": 5,
    "is_weekend": 0,
    "mtd_progress": 0.5,
    "transaction_count": 3
}

print(f"Testing API at {url}...")

for i in range(3):
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        print("✅ Status Code:", response.status_code)
        response_json = response.json()
        print("✅ Response JSON:", json.dumps(response_json, indent=2))
        break
    except Exception as e:
        print(f"⚠️ Attempt {i+1} failed: {e}")
        if i < 2:
            time.sleep(2)
        else:
            print("❌ All attempts to connect to the server failed. Check server.log for errors.")
