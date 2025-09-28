import os
import uuid
import weaviate
import torch
import numpy as np
import stanza
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from weaviate.classes.config import Property, DataType
from weaviate.connect import ConnectionParams
from stanza.pipeline.multilingual import MultilingualPipeline
from SPARQLWrapper import SPARQLWrapper, POST, BASIC, URLENCODED
import datetime
import csv
import torch
from weaviate.connect import ConnectionParams

# --- CONFIGURATION ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOKEN_LIMIT = 8192

# --- MODEL LOADING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

def embed_text(query_text):
    query_vector = model.encode(query_text, task="retrieval.passage")
    return query_vector.tolist()

def import_person(csv_path):
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
    if not client.collections.exists("Persons"):
        client.collections.create(
            name="Persons",
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(name="complement", data_type=DataType.TEXT),
                Property(name="lemma", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT),
                Property(name="birthyear", data_type=DataType.TEXT),
                Property(name="deathyear", data_type=DataType.TEXT),
                Property(name="identifier", data_type=DataType.INT),
                Property(name="url", data_type=DataType.TEXT),
                Property(name="birth_uncertainty", data_type=DataType.BOOL),
                Property(name="death_uncertainty", data_type=DataType.BOOL),
                Property(name="author", data_type=DataType.TEXT),
                Property(name="translator", data_type=DataType.TEXT),
                Property(name="roles", data_type=DataType.TEXT_ARRAY) 
            ],
        )
    count = 1
    with open(csv_path, mode='r', newline='', encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        client.connect()
        for row in reader:
            description = row[8]
            person_id = int(row[0])
            person_name = row[2]+" "+row[1]
            complement = row[2]
            lemma = row[1]
            birthyear = row[3]
            deathyear = row[4]
            url = row[5]
            author = row[9]
            translator = row[10]
            if row[7]: death_uncertainty = True
            else: death_uncertainty = False
            if row[6]: birth_uncertainty = True
            else: birth_uncertainty = False
            desc_vector = embed_text(description)
            client.collections.get("Persons").data.insert(
                    properties={
                        "name": person_name,
                        "hls_id": person_id,
                        "complement": complement,
                        "lemma": lemma,
                        "description": description,
                        "birthyear": birthyear,
                        "deathyear": deathyear,
                        "url": url,
                        "translator": translator,
                        "author": author,
                        "birth_uncertainty":birth_uncertainty,
                        "death_uncertainty": death_uncertainty
                    },
                    vector=desc_vector
                )
            print(f"imported information for {person_name}. N {count}")
            count += 1
        client.close()
if __name__ =="__main__":
    import_person("C:/Users/Silvan/Repo/OCR-Processing-Pipeline/ReferenceData/Historisches_Lexikon/bios.csv")
