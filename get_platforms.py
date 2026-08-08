# get_platforms.py
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("JUICER_API_KEY", "")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Get available platforms
response = requests.get(
    "https://api.juicer.io/v1/platforms",
    headers=headers
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=2))
else:
    print(f"Error: {response.text}")