import os
import requests

TTL_DIR = 'C:/Users/Silvan/Data/ReferenceData/GeoSparql'
HEADERS = {'Content-Type': 'application/x-turtle'}

for filename in os.listdir(TTL_DIR):
    if filename.endswith('.ttl'):
        file_path = os.path.join(TTL_DIR, filename)
        print(f'Uploading {file_path}...')
        with open(file_path, 'rb') as f:
            file_path = os.path.normpath(file_path).replace(os.sep, '/')
            name= file_path.split("/")[-1][:-4]
            print(name)
            endpoint = f"http://localhost:7200/repositories/AIS/rdf-graphs/service?graph=https://lindas.admin.ch/fso/register/{name}"
            response = requests.post(endpoint, headers=HEADERS, data=f)
            if response.status_code == 204:
                print(f"{filename}: Success")
            else:
                print(f"{filename}: Failed ({response.status_code}) - {response.text}")


