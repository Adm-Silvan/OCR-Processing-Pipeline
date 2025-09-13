import os
import requests

TTL_DIR = 'C:/Users/Silvan/Data/AIS/ais_export_part_5135.ttl'
ENDPOINT = "http://localhost:7200/repositories/AIS/statements"
HEADERS = {'Content-Type': 'application/x-turtle'}


with open(TTL_DIR, 'rb') as f:
    response = requests.post(ENDPOINT, headers=HEADERS, data=f)
    if response.status_code == 204:
        print(f"{TTL_DIR}: Success")
    else:
        print(f"{TTL_DIR}: Failed ({response.status_code}) - {response.text}")


