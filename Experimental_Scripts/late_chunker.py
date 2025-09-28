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


# --- CONFIGURATION ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOKEN_LIMIT = 8192
CHUNK_TOKEN_LIMIT = 512
SIMILARITY_THRESHOLD = 0.89
ENDPOINT_URL = "http://localhost:7200/repositories/AIS/statements"
GRAPH = "https://lindas.admin.ch/sfa/ais/tests"
LOG_FILE = f"log_25_09_17.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
# --- WEAVIATE CONNECTION ---
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
CHUNK_COLLECTION = "Chunks"
DOC_COLLECTION = "Documents"
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
            Property(name="file_name", data_type=DataType.TEXT)
        ],
    )


# --- MODEL LOADING ---
print(f"[INFO] Loading tokenizer and model: {MODEL_NAME}")
logging.info(f"Loading tokenizer and model: {MODEL_NAME}")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)
logging.info(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds")
print(f"[INFO] Model loaded in {time.time() - start_time:.2f} seconds")

# --- STANZA SETUP for MULTILINGUAL TOKENIZATION ---
# Download models if necessary (run once)
stanza.download('de')
stanza.download('fr')
stanza.download('it')
nlp = MultilingualPipeline(processors='tokenize')

# Initialize multilingual pipeline with tokenizer only


# --- MERGE FUNCTION WITH CHUNK SIZE LIMIT ---
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
    """Map sentences to token indices"""
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

def store2graph(text,id,order,language): 
    # Define the SPARQL endpoint URL for the repository
    endpoint_url = ENDPOINT_URL

    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setHTTPAuth(BASIC)  # Optional, add if authentication is required
    # sparql.setCredentials("username", "password")  # Uncomment if needed
    sparql.setMethod(POST)
    sparql.setRequestMethod(URLENCODED)
    tailored_text = text.replace("/", "").replace("{", "").replace("}", "").replace("\\", "")
    # Define sample RDF triples in Turtle format to insert
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

def embed_texts(texts):
    # Batch embedding for efficiency
    start = time.time()
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
    duration = time.time() - start
    logging.info(f"Embedding {len(texts)} texts took {duration:.2f} seconds")
    print(f"[INFO] Embedding {len(texts)} texts took {duration:.2f}s")
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

        if chunk_token_count + sent_token_count > token_limit and chunk:
            chunks.append((chunk, chunk_embs))
            chunk, chunk_embs, chunk_token_count = [], [], 0

        if chunk:
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
    """Tokenize text using Stanza pipeline, returns list of sentences (text strings)."""
    doc = nlp(text)
    return [sentence.text for sentence in doc.sentences]


def process_file(file_path, id, signature):
    tracker = EmissionsTracker(save_to_file=True, output_dir="C:/Users/Silvan/Data/Logs", output_file="stats.csv")
    with EmissionsTracker(save_to_file=True, output_dir="C:/Users/Silvan/Data/Logs", output_file="emissions.csv") as tracker:
        total_start = time.time()
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        file_name = os.path.basename(file_path)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=False
        )
        num_tokens = inputs["input_ids"].shape[1]
        print(f"Document '{file_name}' has {num_tokens} tokens.")

        # --- SINGLE WINDOW PROCESSING ---
        if num_tokens <= TOKEN_LIMIT:
            print("Processing with late chunking (single window, variable-length semantic chunks)")
            doc = nlp(text)
            language = doc.lang
            sentences = [sentence.text for sentence in doc.sentences]
            sentence_embeddings = embed_texts(sentences)
            chunked_sentences = semantic_chunking(sentences, sentence_embeddings)
            sentence_token_spans, all_tokens = get_sentence_token_spans(text, sentences)

            # Map chunks to token spans
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
            start = time.time()
            # Generate token embeddings
            with torch.no_grad():
                outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
            token_embeddings = outputs.last_hidden_state.squeeze(0).cpu()
            duration = time.time() - start
            logging.info(f"Late chunking: processed {len(text)} chunks in {duration:.2f} seconds")
            print(f"[INFO] Late chunking: processed {len(text)} chunks in {duration:.2f}s")

            # Store with order
            for order, (chunk, span) in enumerate(merged_data):
                start, end = span
                chunk_text = " ".join(chunk)
                chunk_lang = nlp(chunk_text).lang
                if end > len(token_embeddings):
                    end = len(token_embeddings)
                if end > start:
                    chunk_emb = token_embeddings[start:end].mean(dim=0).to(torch.float32).numpy()
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
                store2graph(chunk_text,id,order,chunk_lang)
            # Store document embedding
            doc_emb = token_embeddings.mean(dim=0).to(torch.float32).numpy()
            client.collections.get(DOC_COLLECTION).data.insert(
                properties={"doc_id": id, "language": language, "file_name": file_name},
                vector=doc_emb.tolist()
            )
            print(f"Stored {len(merged_data)} semantic late chunks and document embedding in Weaviate for '{file_name}'.")

        # --- MULTI-WINDOW PROCESSING ---
        else:
            start = time.time()
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
                start += TOKEN_LIMIT - overlap

            global_chunk_order = 0  # GLOBAL ORDER COUNTER
            doc_embeddings = []

            for w_start, w_end in windows:
                window_tokens = tokens[w_start:w_end]
                window_text = tokenizer.convert_tokens_to_string(window_tokens)
                doc = nlp(window_text)
                language = doc.lang
                window_sentences = [sentence.text for sentence in doc.sentences]
                window_sentence_embeddings = embed_texts(window_sentences)
                chunked_sentences = semantic_chunking(window_sentences, window_sentence_embeddings)

                # Process window
                sentence_token_spans, _ = get_sentence_token_spans(window_text, window_sentences)
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

                # Generate embeddings
                inputs_win = tokenizer(
                    window_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    return_attention_mask=True,
                    truncation=False
                )
                with torch.no_grad():
                    outputs_win = model(**{k: v.to(DEVICE) for k, v in inputs_win.items()})
                token_embeddings = outputs_win.last_hidden_state.squeeze(0).cpu()
                duration = time.time() - start
                logging.info(f"Late chunking: processed {len(window_text)} chunks in {duration:.2f} seconds")
                print(f"[INFO] Late chunking: processed {len(window_text)} chunks in {duration:.2f}s")
                # Store chunks with global order
                for (chunk, span) in merged_data:
                    start, end = span
                    chunk_text = " ".join(chunk)
                    chunk_lang = nlp(chunk_text).lang
                    if end > len(token_embeddings):
                        end = len(token_embeddings)
                    if end > start:
                        chunk_emb = token_embeddings[start:end].mean(dim=0).to(torch.float32).numpy()
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
                        store2graph(chunk_text,id,global_chunk_order,chunk_lang)
                        global_chunk_order += 1

                # Collect window embeddings
                doc_embeddings.append(token_embeddings.mean(dim=0).to(torch.float32).numpy())

            # Store document embedding
            doc_emb = np.mean(doc_embeddings, axis=0)
            client.collections.get(DOC_COLLECTION).data.insert(
                properties={"doc_id": id, "language": language, "file_name": file_name},
                vector=doc_emb.tolist()
            )
            print(f"Stored {global_chunk_order} semantic late chunks and document embedding in Weaviate for '{file_name}'.")
if __name__ == "__main__":
    client.connect()
    process_file("C:/Users/Silvan/Data/OCR_Protocols/1954/05/1954-05-03.txt")
    client.close()