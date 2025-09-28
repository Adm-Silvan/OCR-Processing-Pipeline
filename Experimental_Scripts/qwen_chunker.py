import os
import uuid
import time
import logging
from codecarbon import EmissionsTracker
import weaviate
import torch
import numpy as np
import nltk
from stanza.pipeline.multilingual import MultilingualPipeline
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from weaviate.classes.config import Property, DataType
from weaviate.connect import ConnectionParams
from SPARQLWrapper import SPARQLWrapper, POST, BASIC, URLENCODED

# Download nltk punkt tokenizer resources
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
TOKEN_LIMIT = 32000  # Max context length for Qwen3-4B
CHUNK_TOKEN_LIMIT = 512
SIMILARITY_THRESHOLD = 0.89
ENDPOINT_URL = "http://localhost:7200/repositories/AIS/statements"
GRAPH = "https://lindas.admin.ch/sfa/ais"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE = "processing_log.txt"

# Setup logging
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

# --- Stanza Setup for Language Detection only ---
# Disable all processors except language identification for speed
langid_nlp = MultilingualPipeline(lang_id_config={"langid_clean_text": True}, use_gpu=torch.cuda.is_available())

# --- MODEL LOADING ---
print(f"[INFO] Loading tokenizer and model: {MODEL_NAME}")
logging.info(f"Loading tokenizer and model: {MODEL_NAME}")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
model.to(DEVICE)
logging.info(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds")
print(f"[INFO] Model loaded in {time.time() - start_time:.2f} seconds")

def store2graph(text, id, order, language):
    sparql = SPARQLWrapper(ENDPOINT_URL)
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
        sparql.query()
        logging.info(f"Stored triples for chunk {id}/{order}")
        print(f"[INFO] Stored triples for chunk {id}/{order}")
    except Exception as e:
        logging.error(f"Failed to upload data for chunk {id}/{order}: {e}")
        print(f"[ERROR] Failed to upload data for chunk {id}/{order}: {e}")

def embed_texts(texts):
    start = time.time()
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=CHUNK_TOKEN_LIMIT,
        truncation=True,
        padding=True,
        add_special_tokens=True
    )
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    summed = torch.sum(outputs.last_hidden_state * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    embeddings = (summed / counts).cpu().numpy()
    duration = time.time() - start
    logging.info(f"Embedding {len(texts)} texts took {duration:.2f} seconds")
    print(f"[INFO] Embedding {len(texts)} texts took {duration:.2f}s")
    return embeddings

def late_chunking(text, chunk_token_size):
    print("[INFO] Starting tokenization and full document embedding (late chunking)")
    start = time.time()
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=False)
    input_ids = inputs['input_ids'][0]
    offset_mappings = inputs['offset_mapping'][0]
    with torch.no_grad():
        outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
    token_embs = outputs.last_hidden_state.squeeze(0)
    n_tokens = input_ids.shape[0]
    chunk_spans = [(start, min(start+chunk_token_size, n_tokens)) for start in range(0, n_tokens, chunk_token_size)]
    chunk_texts = []
    chunk_embs = []
    for start, end in chunk_spans:
        char_start = offset_mappings[start][0].item()
        char_end = offset_mappings[end-1][1].item()
        chunk_text = text[char_start:char_end]
        chunk_embedding = token_embs[start:end].mean(dim=0).cpu().numpy()
        chunk_texts.append(chunk_text)
        chunk_embs.append(chunk_embedding)
    duration = time.time() - start
    logging.info(f"Late chunking: processed {len(chunk_texts)} chunks in {duration:.2f} seconds")
    print(f"[INFO] Late chunking: processed {len(chunk_texts)} chunks in {duration:.2f}s")
    return chunk_texts, chunk_embs

def detect_language(text):
    start = time.time()
    doc = langid_nlp(text)
    lang = doc.langid['lang']
    duration = time.time() - start
    logging.info(f"Detected language '{lang}' in {duration:.2f}s")
    print(f"[INFO] Detected language '{lang}' in {duration:.2f}s")
    return lang

def process_file(file_path, id=None, signature=None):
    tracker = EmissionsTracker(save_to_file=True, output_dir="C:/Users/Silvan/Data/Logs", output_file="emissions.csv")
    tracker.start()
    total_start = time.time()

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    file_name = os.path.basename(file_path)
    if not id:
        id = str(uuid.uuid4())

    logging.info(f"Processing document ID: {id}, File: {file_name}, Length (chars): {len(text)}")
    print(f"[INFO] Processing document ID: {id}, File: {file_name}, Length: {len(text)} chars")

    chunk_texts, chunk_embs = late_chunking(text, CHUNK_TOKEN_LIMIT)

    total_tokens = sum(len(tokenizer.tokenize(c)) for c in chunk_texts)
    logging.info(f"Total tokens in document: {total_tokens}")
    print(f"[INFO] Total tokens in document: {total_tokens}")

    global_chunk_order = 0
    for chunk_text, chunk_emb in zip(chunk_texts, chunk_embs):
        chunk_lang = detect_language(chunk_text)
        chunk_id = f"{id}/{global_chunk_order}"

        client.collections.get(CHUNK_COLLECTION).data.insert(
            properties={
                "content": chunk_text,
                "language": chunk_lang,
                "chunk_id": chunk_id,
                "doc_id": id,
                "chunk_order": global_chunk_order,
            },
            vector=chunk_emb.tolist(),
        )
        store2graph(chunk_text, id, global_chunk_order, chunk_lang)
        global_chunk_order += 1

    doc_emb = np.mean(chunk_embs, axis=0)
    doc_lang = detect_language(text)
    client.collections.get(DOC_COLLECTION).data.insert(
        properties={"doc_id": id, "language": doc_lang, "file_name": file_name},
        vector=doc_emb.tolist()
    )

    total_duration = time.time() - total_start
    tracker.stop()
    
    logging.info(f"Finished processing document {id} in {total_duration:.2f} seconds")
    logging.info(f"Estimated CO2 emissions metrics saved")

    print(f"[INFO] Finished processing document {id} in {total_duration:.2f} seconds")
    print(f"[INFO] Estimated CO2 emissions metrics saved to emissions_logs/emissions.csv")

if __name__ == "__main__":
    client.connect()
    process_file("C:/Users/Silvan/Data/OCR_Protocols/1954/05/1954-05-03.txt")
    client.close()
