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

def embed_query(query_text):
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