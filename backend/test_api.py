import requests
import json

url = "http://127.0.0.1:8000/api/predict"
payload = {
    "description": "I was charged twice for the same order.",
    "customer_name": "Test",
    "channel": "Chatbot"
}

try:
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
