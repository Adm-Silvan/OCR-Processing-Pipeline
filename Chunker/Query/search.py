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


def search_chunks(query_text, top_k=5):
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
    collection = client.collections.get("LateChunk")
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
    user_query = "Which text passages are from the month of july and talk about Bern?"
    retrieval_results = search_chunks(user_query, top_k=5)
    print(retrieval_results.generated)
    print(retrieval_results.objects)