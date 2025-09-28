import signal
import requests
import time
import weaviate
from weaviate.connect import ConnectionParams
import os 


def handler(signum, frame):
    global terminate
    print("Termination signal received, exiting gracefully.")
    terminate = True

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

terminate = False
url = "http://localhost:5000/ner"

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
client.connect()
collection = client.collections.use("LoraDocuments")
for obj in collection.iterator(return_properties=["doc_id","file_name"]):
    if terminate: 
        print("finished processing previous document, terminating process")
        break
    doc_id = obj.properties["doc_id"]
    date = obj.properties["file_name"].split(".")[0]
    if os.path.exists('processed_documents.txt'):
        with open('processed_documents.txt') as f:
            if doc_id in f.read():
                continue
    with open("processed_documents.txt", "a") as file:
        file.write(doc_id + "\n")
    data = {
        "doc_id": doc_id,
        "date": date
        }
    print(f"Starting NER for doument {doc_id} with date {date}")
    while True:
        if terminate:
            print("Exiting process once extraction for the document is complete")
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Success:", response.json())
            break
        elif response.status_code == 503:
            print("Still processing...")
            time.sleep(3)
        else:
            print(f"Error {response.status_code}")
            break
    