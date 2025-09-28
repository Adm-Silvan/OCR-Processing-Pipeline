import os
import uuid
import weaviate
import torch
import numpy as np
import stanza
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from weaviate.connect import ConnectionParams
from stanza.pipeline.multilingual import MultilingualPipeline

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



articles = client.collections.use("Persons")
articles_config = articles.config.get()

# Print all property names
for prop in articles_config.properties:
    print(prop.name)

client.close()