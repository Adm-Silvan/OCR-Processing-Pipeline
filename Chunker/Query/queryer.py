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

# --- CONFIGURATION ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOKEN_LIMIT = 8192
CHUNK_TOKEN_LIMIT = 512
SIMILARITY_THRESHOLD = 0.80
# --- MODEL LOADING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)


def stanza_sentence_tokenize(text):
    """Tokenize text using Stanza pipeline, returns list of sentences (text strings)."""
    nlp = MultilingualPipeline(processors='tokenize')
    doc = nlp(text)
    return [sentence.text for sentence in doc.sentences]

def embed_texts(texts):
    # Batch embedding for efficiency
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=CHUNK_TOKEN_LIMIT,
        truncation=True,
        padding=True,
        add_special_tokens=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    summed = torch.sum(outputs.last_hidden_state * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return (summed / counts).cpu().numpy()


def query(query_vector):
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
    result = client.query.get("ChunkClass", ["text"]).with_near_vector({
    "vector": query_vector,
    "certainty": 0.7
    }).with_limit(5).with_additional(["certainty"]).do()
    return result
