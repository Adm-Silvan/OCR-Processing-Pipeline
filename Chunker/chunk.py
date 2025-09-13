import weaviate
import os
import json
from late_chunker import process_file
from weaviate.connect import ConnectionParams
client = weaviate.WeaviateClient(
    connection_params=ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )
)
PATH = "C:/Users/Silvan/Data/OCR_Protocols/"

# load the JSON file
for year in range(1854,1949):
    for month in range(1,13):
        with open(PATH+f"Manifests/{year}-{month}.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        for date, values in data.items():
            id = values[0]
            signature = values[1]
            print(f"Date: {date}, ID: {id},Signature {signature}")
            day = date.split(".")[0]
            if len(str(month)) < 2: month = "0"+str(month)
            if os.path.exists(PATH+f"{year}/{month}/{year}-{month}-{day}.txt"):
                client.connect()
                process_file(PATH+f"{year}/{month}/{year}-{month}-{day}.txt", id, signature) 
                client.close()
            else:
                print(PATH+f"{year}/{month}/{year}-{month}-{day}.txt is missing")
                with open("missing_files.txt", "a") as file:
                    file.write(date + " " + values[0]+ " " + values[1] + "\n")
                    