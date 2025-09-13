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

collection = client.collections.use("Chunks")
filters = Filter.by_property("chunk_id").equal("32329184")

response = collection.query.fetch_objects(filters=filters, limit=75)
for obj in response.objects:
    print(obj.properties)
client.close()