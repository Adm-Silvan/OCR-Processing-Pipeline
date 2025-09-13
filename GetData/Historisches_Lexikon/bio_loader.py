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

# --- SCHEMA CREATION ---
if not client.collections.exists("LateChunk"):
    client.collections.create(
        name="LateChunk",
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="chunk_order", data_type=DataType.INT)  # NEW ORDER FIELD
        ],
    )