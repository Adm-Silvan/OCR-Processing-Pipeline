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
    inputs = tokenizer(
        query_text,
        return_tensors="pt",
        max_length=TOKEN_LIMIT,
        truncation=True,
        add_special_tokens=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  
    
    attn_mask = inputs["attention_mask"].squeeze().unsqueeze(-1).expand(token_embeddings.size()).float()
    attn_mask = attn_mask.to(DEVICE)
    summed = torch.sum(token_embeddings * attn_mask, 0)
    counts = torch.clamp(attn_mask.sum(0), min=1e-9)
    query_vector = (summed / counts).detach().cpu().numpy() 
    return query_vector.tolist()

def import_location(csv_path):
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
    if not client.collections.exists("Place"):
        client.collections.create(
            name="Place",
            properties=[
                Property(name="name_de", data_type=DataType.TEXT),
                Property(name="name_fr", data_type=DataType.TEXT),
                Property(name="name_it", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT),
                Property(name="valid_from", data_type=DataType.TEXT),
                Property(name="identifier", data_type=DataType.TEXT),
                Property(name="url", data_type=DataType.TEXT),
            ],
        )
    count = 1
    with open(csv_path, mode='r', newline='', encoding="utf-8") as csv_file:
        csv_path = os.path.normpath(csv_path).replace(os.sep, '/')
        type_name = csv_path.split("/")[-1][:-4]
        reader = csv.reader(csv_file)
        header = next(reader)
        client.connect()
        for row in reader:
            identifier = row[3]
            url = row[1]
            name_de = row[3]
            desc_vector = embed_text(type_name+" "+name_de)
            client.collections.get("Place").data.insert(
                    properties={
                        "name_de": name_de,
                        "type": type_name,
                        "identifier": identifier,
                        "url": url
                    },
                    vector=desc_vector
                )
            print(f"imported information for {name_de}. N {count}")
            count += 1
        client.close()
if __name__ =="__main__":
    import_location("C:/Users/Silvan/Data/ReferenceData/LINDAS_CSV/Distrikt.csv")
