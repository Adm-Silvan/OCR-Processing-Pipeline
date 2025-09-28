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
from codecarbon import EmissionsTracker
import logging
import datetime
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from weaviate.classes.init import Auth


app = FastAPI()

# Ensure logs directory exists to avoid file errors
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "jinaai/jina-embeddings-v3")
TOKEN_LIMIT = int(os.environ.get("TOKEN_LIMIT", 8192))
CHUNK_TOKEN_LIMIT = int(os.environ.get("CHUNK_TOKEN_LIMIT", 512))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.89))
ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "http://localhost:7200/repositories/AIS/statements")
GRAPH = os.environ.get("GRAPH", "https://lindas.admin.ch/sfa/ais/lora-chunks")
CHUNK_COLLECTION = os.environ.get("CHUNK_COLLECTION", "LoraChunks")
DOC_COLLECTION = os.environ.get("DOC_COLLECTION", "LoraDocuments")
SPARQL_USER = os.environ.get("GRAPHDB_USER")
SPARQL_PASSWORD = os.environ.get("GRAPHDB_PASSWORD")

class IgnoreHttpRequestsFilter(logging.Filter):
    def filter(self, record):
        # Return False to exclude the record from logging if it contains 'HTTP Request:'
        return "HTTP Request:" not in record.getMessage()
# Configure logging to file inside ./logs and also to console
log_file_path = os.path.join(LOG_DIR, "api.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
filter_instance = IgnoreHttpRequestsFilter()
for handler in logging.getLogger().handlers:  # root logger handlers
    handler.addFilter(filter_instance)
class TextInput(BaseModel):
    text: str
    id: Optional[str] = None
    signature: Optional[str] = None
    date: Optional[str] = None

@app.on_event("startup")
def startup_event():
    global client, tokenizer, model, nlp
    # --- CONFIGURATION ---
    global MODEL_NAME, TOKEN_LIMIT, CHUNK_TOKEN_LIMIT, SIMILARITY_THRESHOLD, ENDPOINT_URL, GRAPH, CHUNK_COLLECTION, DOC_COLLECTION
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    logger.info(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds")

    http_host = os.environ.get("WEAVIATE_HTTP_HOST", "localhost")
    http_port = int(os.environ.get("WEAVIATE_HTTP_PORT", 8080))
    http_secure = os.environ.get("WEAVIATE_HTTP_SECURE", "false") == "true"
    grpc_host = os.environ.get("WEAVIATE_GRPC_HOST", "localhost")
    grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", 50051))
    grpc_secure = os.environ.get("WEAVIATE_GRPC_SECURE", "false") == "true"

    api_key = os.environ.get("WEAVIATE_API_KEY")
    user = os.environ.get("WEAVIATE_USER")
    password = os.environ.get("WEAVIATE_PASSWORD")
    bearer_token = os.environ.get("WEAVIATE_BEARER_TOKEN")

    auth = None
    if api_key:
        auth = Auth.api_key(api_key)
    elif user and password:
        auth = Auth.client_password(user, password)
    elif bearer_token:
        auth = Auth.bearer_token(bearer_token)

    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=grpc_secure,
        ),
        auth_client_secret=auth,
    )
    client.connect()

    # Schema creation if not exists
    if not client.collections.exists(CHUNK_COLLECTION):
        client.collections.create(
            name=CHUNK_COLLECTION,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="language", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="chunk_order", data_type=DataType.INT)
            ],
        )

    if not client.collections.exists(DOC_COLLECTION):
        client.collections.create(
            name=DOC_COLLECTION,
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="language", data_type=DataType.TEXT),
                Property(name="file_name", data_type=DataType.TEXT),
                Property(name="token_length", data_type=DataType.INT)
            ],
        )

    # Stanza setup
    stanza.download('de')
    stanza.download('fr')
    stanza.download('it')
    nlp = MultilingualPipeline(processors='tokenize')

model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

def merge_and_limit_chunks(chunked_data, max_tokens):
    """Merge single-line chunks if result stays within max_tokens."""
    merged = []
    for chunk, span in chunked_data:
        chunk_token_count = span[1] - span[0]
        if merged:
            prev_chunk, prev_span = merged[-1]
            prev_token_count = prev_span[1] - prev_span[0]
            combined_token_count = prev_token_count + chunk_token_count
            if len(chunk) == 1 and combined_token_count <= max_tokens:
                merged[-1] = (
                    prev_chunk + chunk,
                    (prev_span[0], span[1])
                )
            else:
                merged.append((chunk, span))
        else:
            merged.append((chunk, span))
    return merged


def get_sentence_token_spans(text, sentences):
    """Map sentences to token indices of the full text."""
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


def store2graph(text, id, order, language):
    endpoint_url = ENDPOINT_URL
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setHTTPAuth(BASIC)
    sparql.setMethod(POST)
    sparql.setRequestMethod(URLENCODED)
    tailored_text = text.replace("/", "").replace("{", "").replace("}", "").replace("\\", "")
    insert_data = f"""
    PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
    INSERT DATA {{
        GRAPH <{GRAPH}> {{
    <https://culture.ld.admin.ch/ais/{id}> rico:WholePartRelation <https://culture.ld.admin.ch/ais/{id}/{order}> .
    <https://culture.ld.admin.ch/ais/{id}/{order}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> rico:RecordPart;
    <https://schema.org/text> '''{tailored_text}'''@{language}.
    }}
    }}
    """
    sparql.setQuery(insert_data)
    try:
        response = sparql.query()
        logging.info(f"Stored triples for chunk {id}/{order}")
        print(f"[INFO] Stored triples for chunk {id}/{order}")
    except Exception as e:
        logging.error(f"Failed to upload data for chunk {id}/{order}: {e}")
        print(f"[ERROR] Failed to upload data for chunk {id}/{order}: {e}")


def semantic_chunking(sentences, sentence_embeddings, token_limit=CHUNK_TOKEN_LIMIT, sim_threshold=SIMILARITY_THRESHOLD):
    """Group sentences into semantically coherent chunks based on sentence embeddings."""
    chunks = []
    chunk = []
    chunk_embs = []
    chunk_token_count = 0

    for i, (sent, emb) in enumerate(zip(sentences, sentence_embeddings)):
        sent_tokens = tokenizer.tokenize(sent)
        sent_token_count = len(sent_tokens)

        # If adding this sentence exceeds token limit, start new chunk
        if chunk_token_count + sent_token_count > token_limit and chunk:
            chunks.append((chunk, chunk_embs))
            chunk, chunk_embs, chunk_token_count = [], [], 0

        if chunk:
            # Compute similarity of current sentence embedding to last sentence embedding in chunk
            sim = cosine_similarity([emb], [chunk_embs[-1]])[0][0]
            if sim < sim_threshold:
                chunks.append((chunk, chunk_embs))
                chunk, chunk_embs, chunk_token_count = [], [], 0

        chunk.append(sent)
        chunk_embs.append(emb)
        chunk_token_count += sent_token_count

    if chunk:
        chunks.append((chunk, chunk_embs))
    return chunks


def stanza_sentence_tokenize(text):
    """Tokenize text using Stanza pipeline, returns list of sentences."""
    doc = nlp(text)
    return [sentence.text for sentence in doc.sentences]


def attention_masked_mean(token_embeddings, attention_mask, start=None, end=None):
    """Compute attention-masked mean for specified span [start:end]."""
    # token_embeddings: (seq_len, hidden_dim)
    # attention_mask: (seq_len,)
    if start is not None and end is not None:
        token_embeddings = token_embeddings[start:end]
        attention_mask = attention_mask[start:end]
    mask = attention_mask.unsqueeze(-1).float()                   # (span_len, 1)
    masked = token_embeddings * mask                              # (span_len, hidden_dim)
    summed = masked.sum(dim=0)
    counts = mask.sum(dim=0).clamp(min=1e-9)
    mean_emb = (summed / counts).to(torch.float32).numpy()
    return mean_emb

@app.post("/process")
def process_file(input: TextInput):
    start_time = time.time()
    text = input.text
    id = input.id if input.id else str(uuid.uuid4())
    file_name = input.date
    try:
        tracker = EmissionsTracker(save_to_file=True, output_dir="/logs", output_file="stats.csv")
        with tracker:
            total_start = time.time()

            inputs = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=False
            )
            num_tokens = inputs["input_ids"].shape[1]
            attention_mask = inputs["attention_mask"].squeeze(0).cpu()    # (seq_len,)

            print(f"Document '{file_name}' has {num_tokens} tokens.")
            logging.info(f"Processing text with {num_tokens} tokens and id {id}")
            if num_tokens <= TOKEN_LIMIT:
                print("Processing with late chunking (single window, using token-span slicing)")

                # Token embeddings for entire document
                with torch.no_grad():
                    outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
                token_embeddings = outputs.last_hidden_state.squeeze(0).cpu() # (seq_len, hidden_dim)

                # Retrieve sentences with stanza
                doc = nlp(text)
                language = doc.lang
                sentences = [sentence.text for sentence in doc.sentences]

                # Get sentence token spans (indices) according to tokenizer tokenization
                sentence_token_spans, _ = get_sentence_token_spans(text, sentences)

                # Generate sentence embeddings by slicing token embeddings using spans and averaging with attention mask
                sentence_embeddings = []
                for start, end in sentence_token_spans:
                    if end > len(token_embeddings):
                        end = len(token_embeddings)
                    if end > start:
                        sent_emb = attention_masked_mean(token_embeddings, attention_mask, start, end)
                    else:
                        sent_emb = np.zeros(token_embeddings.shape[1], dtype=np.float32)
                    sentence_embeddings.append(sent_emb)

                # Semantic chunking on sentence embeddings
                chunked_sentences = semantic_chunking(sentences, sentence_embeddings)

                # Map chunks to token spans using sentence indices
                chunk_token_spans = []
                idx = 0
                for chunk, _ in chunked_sentences:
                    chunk_start = sentence_token_spans[idx][0]
                    chunk_end = sentence_token_spans[idx + len(chunk) - 1][1]
                    chunk_token_spans.append((chunk_start, chunk_end))
                    idx += len(chunk)

                # Merge single-line chunks with length limit
                chunk_data = list(zip([chunk for chunk, _ in chunked_sentences], chunk_token_spans))
                merged_data = merge_and_limit_chunks(chunk_data, CHUNK_TOKEN_LIMIT)

                # Store chunks with embeddings sliced from token embeddings, attention-masked mean
                for order, (chunk, span) in enumerate(merged_data):
                    start_idx, end_idx = span
                    chunk_text = " ".join(chunk)
                    if chunk_text:
                        chunk_lang = nlp(chunk_text).lang
                        if end_idx > len(token_embeddings):
                            end_idx = len(token_embeddings)
                        if end_idx > start_idx:
                            chunk_emb = attention_masked_mean(token_embeddings, attention_mask, start_idx, end_idx)
                            chunk_id = f"{id}/{order}"
                            client.collections.get(CHUNK_COLLECTION).data.insert(
                                properties={
                                    "content": chunk_text,
                                    "language": chunk_lang,
                                    "chunk_id": chunk_id,
                                    "doc_id": id,
                                    "chunk_order": order
                                },
                                vector=chunk_emb.tolist()
                            )
                            store2graph(chunk_text, id, order, chunk_lang)

                # Store document embedding as mean of all token embeddings (attention-masked)
                doc_emb = attention_masked_mean(token_embeddings, attention_mask)
                client.collections.get(DOC_COLLECTION).data.insert(
                    properties={"doc_id": id, "language": language, "file_name": file_name},
                    vector=doc_emb.tolist()
                )
                print(f"Stored {len(merged_data)} semantic late chunks and document embedding in Weaviate for '{file_name}'.")

            else:
                print("Document exceeds token limit, splitting into overlapping windows...")
                tokens = tokenizer.tokenize(text)
                windows = []
                start_token = 0
                overlap = 256
                while start_token < len(tokens):
                    end_token = min(start_token + TOKEN_LIMIT, len(tokens))
                    windows.append((start_token, end_token))
                    if end_token == len(tokens):
                        break
                    start_token += TOKEN_LIMIT - overlap

                global_chunk_order = 0
                doc_embeddings = []

                # Global attention mask for full text remains unchanged
                # token_embeddings for window are local, but spans and mask usage are adjusted below

                for w_start, w_end in windows:
                    window_tokens = tokens[w_start:w_end]
                    window_text = tokenizer.convert_tokens_to_string(window_tokens)

                    # Inputs for window
                    inputs_win = tokenizer(
                        window_text,
                        return_tensors="pt",
                        add_special_tokens=True,
                        return_attention_mask=True,
                        truncation=False
                    )
                    # local attention mask (not used for embedding slicing in chunking below)
                    attention_mask_win = inputs_win["attention_mask"].squeeze(0).cpu()

                    with torch.no_grad():
                        outputs_win = model(**{k: v.to(DEVICE) for k, v in inputs_win.items()})
                    token_embeddings = outputs_win.last_hidden_state.squeeze(0).cpu()

                    doc_win = nlp(window_text)
                    language = doc_win.lang
                    window_sentences = [sentence.text for sentence in doc_win.sentences]

                    # Token spans for sentences in window (local to window_text)
                    sentence_token_spans, _ = get_sentence_token_spans(window_text, window_sentences)
                    # use local spans, local embeddings, local mask for sentence embeddings
                    sentence_embeddings = []
                    for start, end in sentence_token_spans:
                        if end > len(token_embeddings):
                            end = len(token_embeddings)
                        if end > start:
                            sent_emb = attention_masked_mean(token_embeddings, attention_mask_win, start, end)
                        else:
                            sent_emb = np.zeros(token_embeddings.shape[1], dtype=np.float32)
                        sentence_embeddings.append(sent_emb)

                    chunked_sentences = semantic_chunking(window_sentences, sentence_embeddings)

                    chunk_token_spans = []
                    idx = 0
                    for chunk, _ in chunked_sentences:
                        # Map token spans back to global indices by offsetting with w_start
                        chunk_start = sentence_token_spans[idx][0] + w_start
                        chunk_end = sentence_token_spans[idx + len(chunk) - 1][1] + w_start
                        chunk_token_spans.append((chunk_start, chunk_end))
                        idx += len(chunk)

                    chunk_data = list(zip([chunk for chunk, _ in chunked_sentences], chunk_token_spans))
                    merged_data = merge_and_limit_chunks(chunk_data, CHUNK_TOKEN_LIMIT)

                    for chunk, span in merged_data:
                        start_idx, end_idx = span
                        chunk_text = " ".join(chunk)
                        if chunk_text:
                            chunk_lang = nlp(chunk_text).lang
                            # slice embeddings and mask locally relative to window
                            local_start = start_idx - w_start
                            local_end = end_idx - w_start
                            if local_end > len(token_embeddings):
                                local_end = len(token_embeddings)
                            if local_end > local_start >= 0:
                                chunk_emb = attention_masked_mean(token_embeddings, attention_mask_win, local_start, local_end)
                                chunk_id = f"{id}/{global_chunk_order}"
                                client.collections.get(CHUNK_COLLECTION).data.insert(
                                    properties={
                                        "content": chunk_text,
                                        "language": chunk_lang,
                                        "chunk_id": chunk_id,
                                        "doc_id": id,
                                        "chunk_order": global_chunk_order
                                    },
                                    vector=chunk_emb.tolist()
                                )
                                store2graph(chunk_text, id, global_chunk_order, chunk_lang)
                                global_chunk_order += 1

                    # Compute document embedding for this window
                    doc_embeddings.append(attention_masked_mean(token_embeddings, attention_mask_win))

                # Store combined document embedding averaged over all windows
                doc_emb = np.mean(doc_embeddings, axis=0)
                client.collections.get(DOC_COLLECTION).data.insert(
                    properties={"doc_id": id, "language": language, "file_name": file_name},
                    vector=doc_emb.tolist()
                )
                print(f"Stored {global_chunk_order} semantic late chunks and document embedding in Weaviate for '{file_name}'.")
            elapsed = time.time() - start_time
            logging.info(f"Processed input id {id} in {elapsed:.2f} seconds")
        return {"status": "success", "id": id, "processed_tokens": num_tokens}
    except Exception as e:
        logging.error(f"Error processing text id {id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def health_check():
    return {"status": "running"}
