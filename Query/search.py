import weaviate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
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

def embed_query_text(query_text):
    """
    Embeds the query using identical mean pooling (token embeddings averaged)
    as used for chunk storage.
    """
    # Tokenize and get model outputs
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
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # remove .cpu()
    
    # Compute mean pooling (identical to chunk storage)
    attn_mask = inputs["attention_mask"].squeeze().unsqueeze(-1).expand(token_embeddings.size()).float()
    # Ensure attn_mask is on the same device as token_embeddings
    attn_mask = attn_mask.to(DEVICE)
    summed = torch.sum(token_embeddings * attn_mask, 0)
    counts = torch.clamp(attn_mask.sum(0), min=1e-9)
    query_vector = (summed / counts).detach().cpu().numpy()  # move to cpu only at the end
    return query_vector.tolist()


def hybrid_search(query_text, top_k=3):
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
    collection = client.collections.get("SlicedChunks")
    results = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            return_metadata=weaviate.classes.query.MetadataQuery(score=True, explain_score=True),
            alpha=0.25,
            limit=top_k,
            return_properties=["content", "chunk_id", "doc_id", "chunk_order"]
        )
    client.close()
    return results

def vector_search(query_text, top_k=3):
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
    collection = client.collections.get("SlicedChunks")
    results = collection.query.near_vector(
            near_vector=query_vector,
            return_metadata=weaviate.classes.query.MetadataQuery(certainty=True),
            limit=top_k,
            return_properties=["content", "chunk_id", "doc_id", "chunk_order"]
        )
    client.close()
    return results



if __name__ == "__main__":
    # Example usage:
    user_query = "Was wurde im Februar 1913 zu Deutschland besprochen?"
    print("QUERY: " + user_query)
    retrieval_results = hybrid_search(user_query, top_k=5)
    print("\nHYBRID SEARCH RESULTS\n-------------------------------------------------")
    for o in retrieval_results.objects:
         print(o.properties)
    print("\nVECTOR SEARCH RESULTS\n-------------------------------------------------")
    user_query = "Was wurde im Februar 1913 zu Deutschland besprochen?"
    retrieval_results = vector_search(user_query, top_k=5)
    for o in retrieval_results.objects:
        print(o.properties)