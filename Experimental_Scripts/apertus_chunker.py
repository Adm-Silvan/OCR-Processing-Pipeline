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


# --- CONFIGURATION ---

hf_token = "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"
MODEL_NAME = "swiss-ai/Apertus-8B-2509"
TOKEN_LIMIT = 2000  # Reduced from 8192
CHUNK_TOKEN_LIMIT = 512
SIMILARITY_THRESHOLD = 0.89
ENDPOINT_URL = "http://localhost:7200/repositories/AIS/statements"
GRAPH = "https://lindas.admin.ch/sfa/ais"

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
CHUNK_COLLECTION = "Apertus_Chunks"
DOC_COLLECTION = "Apertus_Documents"
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, use_auth_token=hf_token, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)


# --- STANZA SETUP for MULTILINGUAL TOKENIZATION ---
stanza.download('de')
stanza.download('fr')
stanza.download('it')
nlp = MultilingualPipeline(processors='tokenize')


def merge_and_limit_chunks(chunked_data, max_tokens):
    """Merge single-line chunks if combined token count is within limits."""
    merged = []
    for chunk, span in chunked_data:
        chunk_token_count = span[1] - span[0]
        if merged:
            prev_chunk, prev_span = merged[-1]
            prev_token_count = prev_span[1] - prev_span[0]
            combined_token_count = prev_token_count + chunk_token_count
            if len(chunk) == 1 and combined_token_count <= max_tokens:
                merged[-1] = (prev_chunk + chunk, (prev_span[0], span[1]))
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
        sparql.query()
        print(f"Stored triples for chunk {id}/{order}")
    except Exception as e:
        print("Failed to upload data:", e)


def embed_texts(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=CHUNK_TOKEN_LIMIT,
        truncation=True,
        padding=True,
        add_special_tokens=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    summed = torch.sum(outputs.last_hidden_state * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return (summed / counts).cpu().numpy()


def semantic_chunking(sentences, sentence_embeddings, token_limit=CHUNK_TOKEN_LIMIT, sim_threshold=SIMILARITY_THRESHOLD):
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
    doc = nlp(text)
    return [sentence.text for sentence in doc.sentences]


def process_file(file_path, id=None, signature=None):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    file_name = os.path.basename(file_path)
    if not id:
        id = str(uuid.uuid4())

    # Tokenize entire document
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=False,
    )
    num_tokens = inputs["input_ids"].shape[1]
    print(f"Document '{file_name}' has {num_tokens} tokens.")

    if num_tokens <= TOKEN_LIMIT:
        print("Processing with late chunking (single window, variable-length semantic chunks)")
        doc = nlp(text)
        language = doc.lang
        sentences = [sentence.text for sentence in doc.sentences]
        sentence_embeddings = embed_texts(sentences)
        chunked_sentences = semantic_chunking(sentences, sentence_embeddings)
        sentence_token_spans, all_tokens = get_sentence_token_spans(text, sentences)

        chunk_token_spans = []
        idx = 0
        for chunk, _ in chunked_sentences:
            chunk_start = sentence_token_spans[idx][0]
            chunk_end = sentence_token_spans[idx + len(chunk) - 1][1]
            chunk_token_spans.append((chunk_start, chunk_end))
            idx += len(chunk)

        chunk_data = list(zip([chunk for chunk, _ in chunked_sentences], chunk_token_spans))
        merged_data = merge_and_limit_chunks(chunk_data, CHUNK_TOKEN_LIMIT)

        with torch.no_grad():
            outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
        token_embeddings = outputs.last_hidden_state.squeeze(0).cpu()

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
                        "chunk_order": order,
                    },
                    vector=chunk_emb.tolist(),
                )
                store2graph(chunk_text, id, order, chunk_lang)
        doc_emb = token_embeddings.mean(dim=0).to(torch.float32).numpy()
        client.collections.get(DOC_COLLECTION).data.insert(
            properties={"doc_id": id, "language": language, "file_name": file_name}, vector=doc_emb.tolist()
        )
        print(f"Stored {len(merged_data)} semantic late chunks and document embedding in Weaviate for '{file_name}'.")

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
            start += TOKEN_LIMIT - overlap

        global_chunk_order = 0
        doc_embeddings = []

        for w_start, w_end in windows:
            window_tokens = tokens[w_start:w_end]
            window_text = tokenizer.convert_tokens_to_string(window_tokens)
            doc = nlp(window_text)
            language = doc.lang
            window_sentences = [sentence.text for sentence in doc.sentences]
            window_sentence_embeddings = embed_texts(window_sentences)
            chunked_sentences = semantic_chunking(window_sentences, window_sentence_embeddings)

            sentence_token_spans, _ = get_sentence_token_spans(window_text, window_sentences)
            chunk_token_spans = []
            idx = 0
            for chunk, _ in chunked_sentences:
                chunk_start = sentence_token_spans[idx][0]
                chunk_end = sentence_token_spans[idx + len(chunk) - 1][1]
                chunk_token_spans.append((chunk_start, chunk_end))
                idx += len(chunk)

            chunk_data = list(zip([chunk for chunk, _ in chunked_sentences], chunk_token_spans))
            merged_data = merge_and_limit_chunks(chunk_data, CHUNK_TOKEN_LIMIT)

            inputs_win = tokenizer(
                window_text,
                return_tensors="pt",
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=False,
            )
            with torch.no_grad():
                outputs_win = model(**{k: v.to(DEVICE) for k, v in inputs_win.items()})
            token_embeddings = outputs_win.last_hidden_state.squeeze(0).cpu()

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
                            "chunk_order": global_chunk_order,
                        },
                        vector=chunk_emb.tolist(),
                    )
                    store2graph(chunk_text, id, global_chunk_order, chunk_lang)
                    global_chunk_order += 1

            doc_embeddings.append(token_embeddings.mean(dim=0).to(torch.float32).numpy())

        doc_emb = np.mean(doc_embeddings, axis=0)
        client.collections.get(DOC_COLLECTION).data.insert(
            properties={"doc_id": id, "language": language, "file_name": file_name}, vector=doc_emb.tolist()
        )
        print(f"Stored {global_chunk_order} semantic late chunks and document embedding in Weaviate for '{file_name}'.")


if __name__ == "__main__":
    client.connect()
    process_file("C:/Users/Silvan/Data/OCR_Protocols/1954/05/1954-05-03.txt")
    client.close()
