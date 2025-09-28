import weaviate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from weaviate.connect import ConnectionParams
from weaviate.classes.query import Filter


# --- CONFIGURATION ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOKEN_LIMIT = 8192

# --- MODEL LOADING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

def embed_query_text(query_text):
    query_vector = model.encode(query_text, task="retrieval.query")
    return query_vector.tolist()

def hybrid_search(doc_ids,query_text, query_vector, top_k=3):
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
    filters=Filter.by_property("doc_id").contains_any(doc_ids)    
    collection = client.collections.get("LoraChunks")
    results = collection.query.hybrid(
            filters=filters,
            query=query_text,
            vector=query_vector,
            return_metadata=weaviate.classes.query.MetadataQuery(score=True, explain_score=True),
            alpha=0.25,
            limit=top_k,
            return_properties=["content", "chunk_id", "doc_id", "chunk_order"]
        )
    client.close()
    return results

def vector_search(doc_ids, query_vector, top_k=3):
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
    filters=Filter.by_property("doc_id").contains_any(doc_ids)
    collection = client.collections.get("LoraChunks")
    results = collection.query.near_vector(
            filters=filters,
            near_vector=query_vector,
            return_metadata=weaviate.classes.query.MetadataQuery(certainty=True),
            limit=top_k,
            return_properties=["content", "chunk_id", "doc_id", "chunk_order"]
        )
    client.close()
    return results

def doc_search(query_text, top_k=10):
    query_vector = embed_query_text(query_text)

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
    collection = client.collections.get("LoraDocuments")
    doc_results = collection.query.near_vector(
            near_vector=query_vector,
            return_metadata=weaviate.classes.query.MetadataQuery(certainty=True),
            limit=top_k,
            return_properties=["doc_id", "file_name"]
        )
    client.close()
    doc_ids = []
    print("The following documents were matched:")
    for o in doc_results.objects:
        doc_ids.append(o.properties['doc_id'])
        print(o)
    results = []
    results.append(vector_search(doc_ids,query_vector))
    results.append(hybrid_search(doc_ids,query_text,query_vector))
    return results



if __name__ == "__main__":
    # Example usage:
    user_query = "What were the major controversies or debates reflected in the Federal Council minutes from 1908 to 1913?"
    print("QUERY: " + user_query)
    print("\nVECTOR SEARCH RESULTS\n-------------------------------------------------")
    results = doc_search(user_query, top_k=5)
    for retrieval_results in results: 
        for o in retrieval_results.objects:
            print(o)