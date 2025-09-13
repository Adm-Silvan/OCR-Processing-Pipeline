import os
import uuid
import weaviate
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')

# --- CONFIGURATION ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOKEN_LIMIT = 8192
CHUNK_TOKEN_LIMIT = 512  # Max tokens per chunk
SIMILARITY_THRESHOLD = 0.80  # Cosine similarity threshold for semantic grouping

# --- CONNECT TO WEAVIATE ---
client = weaviate.WeaviateClient(
    connection_params=weaviate.connect.ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )
)
client.connect()
# --- CREATE WEAVIATE SCHEMA IF NEEDED ---
# Schema creation
if not client.collections.exists("LateChunk"):
    client.collections.create(
        name="LateChunk",
        properties=[
            weaviate.classes.config.Property(name="content", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="chunk_id", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="doc_id", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="chunk_order", data_type=weaviate.classes.config.DataType.INT)
        ],
    )

if not client.collections.exists("DocumentEmbedding"):
    client.collections.create(
        name="DocumentEmbedding",
        properties=[
            weaviate.classes.config.Property(name="doc_id", data_type=weaviate.classes.config.DataType.TEXT),
            weaviate.classes.config.Property(name="file_name", data_type=weaviate.classes.config.DataType.TEXT)
        ],
    )


# --- LOAD MODEL AND TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

def merge_single_line_chunks(chunked_data):
    """Merge single-line chunks into previous chunk"""
    merged = []
    for chunk, span in chunked_data:
        if len(chunk) == 1 and merged:
            # Merge with previous chunk
            prev_chunk, prev_span = merged[-1]
            merged[-1] = (
                prev_chunk + chunk,
                (prev_span[0], span[1])
            )
        else:
            merged.append((chunk, span))
    return merged

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
        outputs = model(**inputs, task="retrieval.passage")
    # Mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    summed = torch.sum(outputs.last_hidden_state * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return (summed / counts).cpu().numpy()

def semantic_chunking(sentences, sentence_embeddings, token_limit=CHUNK_TOKEN_LIMIT, sim_threshold=SIMILARITY_THRESHOLD):
    """Group sentences into semantically coherent chunks."""
    chunks = []
    chunk = []
    chunk_embs = []
    chunk_token_count = 0

    for i, (sent, emb) in enumerate(zip(sentences, sentence_embeddings)):
        sent_tokens = tokenizer.tokenize(sent)
        sent_token_count = len(sent_tokens)
        # If adding this sentence would exceed chunk token limit, start new chunk
        if chunk_token_count + sent_token_count > token_limit and chunk:
            chunks.append((chunk, chunk_embs))
            chunk, chunk_embs, chunk_token_count = [], [], 0

        # If not first sentence in chunk, check semantic similarity with previous
        if chunk:
            sim = cosine_similarity([emb], [chunk_embs[-1]])[0][0]
            if sim < sim_threshold:
                # Semantic break, start new chunk
                chunks.append((chunk, chunk_embs))
                chunk, chunk_embs, chunk_token_count = [], [], 0

        chunk.append(sent)
        chunk_embs.append(emb)
        chunk_token_count += sent_token_count

    # Add remaining chunk
    if chunk:
        chunks.append((chunk, chunk_embs))
    return chunks

def get_sentence_token_spans(text, sentences):
    """Return list of (start_token_idx, end_token_idx) for each sentence."""
    spans = []
    tokens = []
    idx = 0
    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)
        start = idx
        end = idx + len(sent_tokens)
        spans.append((start, end))
        tokens.extend(sent_tokens)
        idx = end
    return spans, tokens

def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    file_name = os.path.basename(file_path)
    doc_id = str(uuid.uuid4())

    # Tokenize full document
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=False
    )
    num_tokens = inputs["input_ids"].shape[1]
    print(f"Document '{file_name}' has {num_tokens} tokens.")

    # Split into sentences
    sentences = sent_tokenize(text, language="german")
    sentence_embeddings = embed_texts(sentences)
    sentence_token_spans, all_tokens = get_sentence_token_spans(text, sentences)

    # --- CASE 1: Document fits in one window ---
    if num_tokens <= TOKEN_LIMIT:
        print("Processing with late chunking (single window, variable-length semantic chunks)...")
        # 1. Group sentences into semantic chunks
        chunked_sentences = semantic_chunking(sentences, sentence_embeddings)
        # 2. Map chunk boundaries to token spans
        chunk_token_spans = []
        idx = 0
        for chunk, _ in chunked_sentences:
            chunk_start = sentence_token_spans[idx][0]
            chunk_end = sentence_token_spans[idx + len(chunk) - 1][1]
            chunk_token_spans.append((chunk_start, chunk_end))
            idx += len(chunk)
        # 3. Get token-level embeddings for the whole document
        with torch.no_grad():
            outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()}, task="retrieval.passage")
        token_embeddings = outputs.last_hidden_state.squeeze(0).cpu()  # [num_tokens, hidden_size]
        # 4. Pool token embeddings for each chunk
        for (chunk, _), (start, end) in zip(chunked_sentences, chunk_token_spans):
            chunk_text = " ".join(chunk)
            if end > len(token_embeddings):
                end = len(token_embeddings)
            if end > start:
                chunk_emb = token_embeddings[start:end].mean(dim=0).numpy()
                chunk_id = str(uuid.uuid4())
                client.collections.get("LateChunk").data.insert(
                properties={"content": chunk_text, "chunk_id": chunk_id, "doc_id": doc_id},
                vector=chunk_emb.tolist()
                )
        # Store full document embedding
        doc_emb = token_embeddings.mean(dim=0).numpy()
        client.collections.get("DocumentEmbedding").data.insert(
            properties={"doc_id": doc_id, "file_name": file_name},
            vector=doc_emb.tolist()
        )
        print(f"Stored {len(chunked_sentences)} semantic late chunks and document embedding in Weaviate for '{file_name}'.")
    # --- CASE 2: Document needs windowing ---
    else:
        print("Document exceeds token limit, splitting into overlapping windows...")
        tokens = tokenizer.tokenize(text)
        windows = []
        start = 0
        overlap = 256
        while start < len(tokens):
            end = min(start + TOKEN_LIMIT, len(tokens))
            windows.append((start, end))
            if end == len(tokens):
                break
            start += TOKEN_LIMIT - overlap  # Overlap
        all_chunk_count = 0
        doc_embeddings = []
        for w_idx, (w_start, w_end) in enumerate(windows):
            window_tokens = tokens[w_start:w_end]
            window_text = tokenizer.convert_tokens_to_string(window_tokens)
            window_sentences = sent_tokenize(window_text, language="german")
            window_sentence_embeddings = embed_texts(window_sentences)
            window_sentence_token_spans, _ = get_sentence_token_spans(window_text, window_sentences)
            # 1. Group sentences into semantic chunks
            chunked_sentences = semantic_chunking(window_sentences, window_sentence_embeddings)
            # 2. Map chunk boundaries to token spans (relative to window)
            chunk_token_spans = []
            idx = 0
            for chunk, _ in chunked_sentences:
                chunk_start = window_sentence_token_spans[idx][0]
                chunk_end = window_sentence_token_spans[idx + len(chunk) - 1][1]
                chunk_token_spans.append((chunk_start, chunk_end))
                idx += len(chunk)
            # 3. Get token-level embeddings for the window
            inputs_win = tokenizer(
                window_text,
                return_tensors="pt",
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=False
            )
            with torch.no_grad():
                outputs_win = model(**{k: v.to(DEVICE) for k, v in inputs_win.items()}, task="retrieval.passage")
            token_embeddings = outputs_win.last_hidden_state.squeeze(0).cpu()
            # 4. Pool token embeddings for each chunk
            for (chunk, _), (start, end) in zip(chunked_sentences, chunk_token_spans):
                chunk_text = " ".join(chunk)
                if end > len(token_embeddings):
                    end = len(token_embeddings)
                if end > start:
                    chunk_emb = token_embeddings[start:end].mean(dim=0).numpy()
                    chunk_id = str(uuid.uuid4())
                    client.collections.get("LateChunk").data.insert(
                    properties={"content": chunk_text, "chunk_id": chunk_id, "doc_id": doc_id},
                    vector=chunk_emb.tolist()
                    )
                    all_chunk_count += 1
            # Store window embedding for doc embedding pooling
            doc_embeddings.append(token_embeddings.mean(dim=0).numpy())
        # Store document-level embedding (mean of all window embeddings)
        doc_emb = np.mean(doc_embeddings, axis=0)

        client.collections.get("DocumentEmbedding").data.insert(
            properties={"doc_id": doc_id, "file_name": file_name},
            vector=doc_emb.tolist()
        )

        print(f"Stored {all_chunk_count} semantic late chunks and document embedding in Weaviate for '{file_name}'.")
client.close()
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    client.connect()
    process_file("C:/Data/OCR_Protocols/1848/11/1848-11-21.txt")
    client.close()
    """ import sys
    if len(sys.argv) < 2:
        print("Usage: python late_semantic_chunking_jina_de.py <file1.txt> [file2.txt ...]")
        exit(1)
    for file_path in sys.argv[1:]:
        print(f"\nProcessing file: {file_path}")
        process_file(file_path) """
