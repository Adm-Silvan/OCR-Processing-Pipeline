import os
import requests

TTL_DIR = 'C:/Users/Silvan/Data/AIS'
ENDPOINT = "http://localhost:7200/repositories/AIS/statements"
HEADERS = {'Content-Type': 'application/x-turtle'}

for filename in os.listdir(TTL_DIR):
    if filename.endswith('.ttl'):
        file_path = os.path.join(TTL_DIR, filename)
        print(f'Uploading {file_path}...')
        with open(file_path, 'rb') as f:
            response = requests.post(ENDPOINT, headers=HEADERS, data=f)
            if response.status_code == 204:
                print(f"{filename}: Success")
            else:
                print(f"{filename}: Failed ({response.status_code}) - {response.text}")


