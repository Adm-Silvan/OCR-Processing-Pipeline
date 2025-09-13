import spacy
from datetime import datetime, timedelta
import weaviate
import torch
from transformers import AutoTokenizer, AutoModel
from weaviate.classes.query import Filter
from weaviate.connect import ConnectionParams
from thefuzz import fuzz

""" 
# --- CONFIGURATION ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOKEN_LIMIT = 8192

# --- MODEL LOADING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

def embed_text(query_text):
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
    token_embeddings = outputs.last_hidden_state.squeeze(0)  
    
    attn_mask = inputs["attention_mask"].squeeze().unsqueeze(-1).expand(token_embeddings.size()).float()
    attn_mask = attn_mask.to(DEVICE)
    summed = torch.sum(token_embeddings * attn_mask, 0)
    counts = torch.clamp(attn_mask.sum(0), min=1e-9)
    query_vector = (summed / counts).detach().cpu().numpy() 
    return query_vector.tolist()
 """

def extract_entities(nlp,text):
    """Extract people, organizations, and places from text using spaCy NER."""
    doc = nlp(text)
    entities = {"PER": set(), "ORG": set(), "GPE": set()}
    for ent in doc.ents:
        if ent.label_ in entities:
            # Normalize entity text by stripping whitespace
            entities[ent.label_].add(ent.text.strip())
    return entities

def query_weaviate(client, collection_name, entity_names, text_vector, date=None, top_k=5):
    """
    Query Weaviate hybrid search per entity type.
    For 'People' collection, filter out entities with deathyear > 10 years older than date.
    """
    results = []
    collection = client.collections.get(collection_name)

    for name in entity_names:
        query_args = {
            "query": name,
            "vector": text_vector,
            "alpha": 0.25,
            "limit": top_k,
            "return_metadata": weaviate.classes.query.MetadataQuery(score=True, explain_score=True),
            "return_properties": ["name", "hls_id", "url"]
        }
        if collection_name == "People":
            query_args["return_properties"].append("deathyear")

        response = collection.query.hybrid(**query_args)
        for match in response.objects:
            props = match.properties
            candidate_name = props.get("name", "")
            # Fuzzy string matching between recognized name and candidate name
            fuzzy_score = fuzz.token_sort_ratio(name.lower(), candidate_name.lower())

            if fuzzy_score < FUZZ:
                # Skip candidates that don't meet fuzzy matching threshold
                continue
            # For people, filter by deathyear vs document date
            if collection_name == "People" and "deathyear" in props and date:
                deathyear_str = props.get("deathyear")
                if deathyear_str:
                    try:
                        deathyear = datetime(int(deathyear_str.split(".")[0]), 1,1)
                        if deathyear < date - timedelta(days=3650):  # 10 years before date
                            continue  # Skip old death date
                    except Exception:
                        print("failed")
                        pass  # If date parsing fails, don't filter out
            # Check confidence/certainty score if available, include matches with high confidence
            if match.metadata.score > CERTAINTY:
                results.append({
                    "name": props.get("name", name),
                    "id": props.get("hls_id"),
                    "url": props.get("url"),
                    "deathyear": props.get("deathyear") if collection_name == "People" else None,
                    "score": match.metadata.score, 
                    "fuzzy_score": fuzzy_score
                })
    return results

def get_docinfo(file_name):
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
    collection = client.collections.use("Documents")
    filters = Filter.by_property("file_name").equal(file_name)
    response = collection.query.fetch_objects(filters=filters, limit=1,include_vector=True)
    for obj in response.objects:
        doc_id = obj.properties["doc_id"]
        vector = obj.vector["default"]
    collection = client.collections.use("Chunks")
    filters = Filter.by_property("chunk_id").equal(doc_id)
    response = collection.query.fetch_objects(filters=filters, limit=1000)
    languages = set()
    chunks = []
    for obj in response.objects:
        languages.add(obj.properties["language"])
        chunks.append(obj.properties["chunk_id"])
    client.close()
    return doc_id, vector, languages, chunks

def disambiguate(nlp, text, text_vector, text_date, client, top_k=5):
    # Parse date string to datetime
    if isinstance(text_date, str):
        text_date = datetime.strptime(text_date, "%Y-%m-%d")
    entities = extract_entities(nlp,text)
    # Query Weaviate by entity type/collection
    people_results = query_weaviate(client, "People", entities["PER"], text_vector, text_date, top_k)
    #org_results = query_weaviate(client, "Organizations", entities["ORG"], text_vector, None, top_k)
    #place_results = query_weaviate(client, "Places", entities["GPE"], text_vector, None, top_k)

    return {
        "people": people_results}
        #"organizations": org_results,
        #"places": place_results
    
def combine_text_chunks(segments):
    # Convert list of single-key dicts into a list of (order, text) tuples
    order_text_pairs = [(list(d.keys())[0], list(d.values())[0]) for d in segments]
    
    # Sort by the order key
    order_text_pairs.sort(key=lambda x: x[0])
    
    combined_chunks = []
    current_chunk = []
    previous_order = None
    
    for order, text in order_text_pairs:
        if previous_order is not None and order != previous_order + 1:
            # Gap detected, start new chunk
            combined_chunks.append(" ".join(current_chunk))
            current_chunk = []
        
        current_chunk.append(text)
        previous_order = order
    
    # Add the final chunk
    if current_chunk:
        combined_chunks.append(" ".join(current_chunk))
    
    return combined_chunks

def main(file,date):
    file_name = file.split("/")[-1]
    doc_id, vector, languages, chunks = get_docinfo(file_name)
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
    if len(languages) > 1: 
        collection = client.collections.use("Chunks")
        de_chunks = []
        fr_chunks = []
        it_chunks = []
        other_chunks = []
        results = []
        for chunk in chunks:
            filters = Filter.by_property("chunk_id").equal(chunk)
            response = collection.query.fetch_objects(filters=filters, limit=1)
            for obj in response.objects:
                lang = obj.properties["language"]
                text = obj.properties["content"]
                order = obj.properties["chunk_order"]
                if lang == "de":
                    de_chunks.append({order:text})
                elif lang == "fr":
                    fr_chunks.append({order:text})
                elif lang == "it":
                    it_chunks.append({order:text})
                else:
                    other_chunks.append({order:text})
        if de_chunks:
            de_text = combine_text_chunks(de_chunks)
            nlp = spacy.load("de_core_news_lg")
            for text in de_text:
                results.append(disambiguate(nlp,text,vector,date,client))
        if fr_chunks:
            fr_text = combine_text_chunks(fr_chunks)
            nlp = spacy.load("fr_core_news_lg")
            for text in fr_text:
                results.append(disambiguate(nlp,text,vector,date,client))        
        if it_chunks:
            it_text = combine_text_chunks(it_chunks)
            nlp = spacy.load("it_core_news_lg")
            for text in it_text:
                results.append(disambiguate(nlp,text,vector,date,client))
        if other_chunks:
            other_text = combine_text_chunks(other_chunks)
            nlp = spacy.load("xx_ent_wiki_sm")
            for text in other_text:
                results.append(disambiguate(nlp,text,vector,date,client))
        combined = {
            "people": [],
            "organizations": [],
            "places": []
        }

        for result in results:
            combined["people"].extend(result.get("people", []))
            combined["organizations"].extend(result.get("organizations", []))
            combined["places"].extend(result.get("places", []))
        client.close()
        return combined
    else:
        for lang in languages:
            if lang == "de": nlp = spacy.load("de_core_news_lg")
            elif lang == "fr": nlp = spacy.load("fr_core_news_lg")
            elif lang == "it": nlp = spacy.load("it_core_news_lg")
            else: print(f"Something went wrong retrieving language information. Got value: {lang}")
        with open(file, 'r') as text_file:
            text = text_file.read()
        client.close()
        return disambiguate(nlp,text,vector,date,client)
CERTAINTY = 0.7
FUZZ = 70
if __name__ == "__main__":
    file="C:/Users/Silvan/Data/OCR_Protocols/1957/05/1957-05-21.txt"
    date = file.split("/")[-1][:-4]
    print(date)
    print(main(file,date)) 
# Example usage:
# Assume you have authenticated Weaviate client
# client = Client("http://localhost:8080")

# textchunk = "Barack Obama met with Microsoft representatives in New York."
# textchunk_vector = [...]  # Vector embedding for textchunk
# text_date = "2023-05-01"

# results = main(textchunk, textchunk_vector, text_date, client)
# print(results)