import spacy
from datetime import datetime, timedelta
import weaviate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from weaviate.classes.query import Filter
from weaviate.connect import ConnectionParams
from thefuzz import fuzz
import os
import ast
from SPARQLWrapper import SPARQLWrapper, POST, BASIC, URLENCODED
from codecarbon import EmissionsTracker
import logging
from apertus_get_roles import  get_roles_apertus
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lora_query import embed_lora_query
from weaviate.classes.init import Auth

app = FastAPI()

# Ensure logs directory exists to avoid file errors
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

CERTAINTY = float(os.environ.get("CERTAINTY", 0.85))
FUZZ = int(os.environ.get("FUZZ", 85))
ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "http://localhost:7200/repositories/AIS/statements")
GRAPH = os.environ.get("GRAPH", "https://lindas.admin.ch/sfa/ais/checked_entities")
CHUNK_COLLECTION = os.environ.get("CHUNK_COLLECTION", "LoraChunks")
DOC_COLLECTION = os.environ.get("DOC_COLLECTION", "LoraDocuments")
PEOPLE_COLLECTION = os.environ.get("PEOPLE_COLLECTION", "Persons")
ORG_COLLECTION = os.environ.get("ORG_COLLECTION", "Organizations")
GEO_COLLECTION = os.environ.get("GEO_COLLECTION", "Places")
SPARQL_USER = os.environ.get("GRAPHDB_USER")
SPARQL_PASSWORD = os.environ.get("GRAPHDB_PASSWORD")

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

global client

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



log_file_path = os.path.join(LOG_DIR, "ner_api.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
class TextInput(BaseModel):
    doc_id: str
    date: str

def extract_last_name(name):
    parts = name.strip().split()
    return parts[-1] if len(parts) > 1 else parts[0]

def group_and_reduce_entities(entities, threshold=85):
    full_names = [e for e in entities if len(e.strip().split()) > 1]
    single_names = [e for e in entities if len(e.strip().split()) == 1]
    
    reduced_set = set(full_names)
    for single in single_names:
        matched = False
        for full in full_names:
            # Fuzzy match single name to last name in full name
            last_name = extract_last_name(full)
            score = fuzz.ratio(single.lower(), last_name.lower())
            if score >= threshold:
                matched = True
                break
        if not matched:
            # It's not a redundant mention, treat as standalone entity
            reduced_set.add(single)
    # Only return full names or unmatched single names
    return sorted(reduced_set)

def extract_entities(nlp,text):
    """Extract people, organizations, and places from text using spaCy NER."""
    doc = nlp(text)
    entities = {"PER": set(), "ORG": set(), "LOC": set()}
    for ent in doc.ents:
        if ent.label_ in entities:
            # Normalize entity text by stripping whitespace
            entities[ent.label_].add(ent.text.strip())
    entities["PER"] = group_and_reduce_entities(list(entities["PER"]), threshold=85)
    print(f"Spacy identified {len(entities.get("PER"))} people, {len(entities.get("ORG"))} organizations and {len(entities.get("LOC"))} places as candidate entities")
    logging.info(f"Spacy identified {len(entities.get("PER"))} people, {len(entities.get("ORG"))} organizations and {len(entities.get("LOC"))} places as candidate entities")
    return entities

def query_weaviate(client, collection_name, entity_names, text_vector, language, date=None, top_k=5):
    """
    Query Weaviate hybrid search per entity type.
    For 'People' collection, filter out entities with deathyear > 10 years older than date.
    """
    results = []
    client.connect()
    collection = client.collections.get(collection_name)

    for name in entity_names:
        if collection_name == GEO_COLLECTION or collection_name == ORG_COLLECTION:
            text_vector = embed_lora_query(name)
        query_args = {
            "query": name,
            "vector": text_vector,
            "alpha": 0.25,
            "limit": top_k,
            "return_metadata": weaviate.classes.query.MetadataQuery(score=True, explain_score=True),
            "return_properties": ["url"]
        }
        if collection_name == PEOPLE_COLLECTION:
            query_args["return_properties"].extend(["name", "description", "identifier", "deathyear"])
        if collection_name == GEO_COLLECTION:
            query_args["return_properties"].extend(["name_de", "name_fr", "name_it", "type", "identifier", ])
        if collection_name == ORG_COLLECTION:
            query_args["return_properties"].extend(["name_de", "name_fr"])

        response = collection.query.hybrid(**query_args)
        for match in response.objects:
            props = match.properties
            if collection_name == PEOPLE_COLLECTION:
                candidate_name = props.get("name", "")
            elif collection_name == ORG_COLLECTION:
                if language == "fr":
                    candidate_name = props.get("name_fr", "")
                else: candidate_name = props.get("name_de", "")
            elif props.get("type") == "District":
                candidate_name = props.get("name_de", "")
            else: 
                candidate_name = props.get(f"name_{language}")
            # Fuzzy string matching between recognized name and candidate name
            fuzzy_score = fuzz.token_sort_ratio(name.lower(), candidate_name.lower())

            if fuzzy_score < FUZZ:
                # Skip candidates that don't meet fuzzy matching threshold
                continue
            # For people, filter by deathyear vs document date
            if collection_name == PEOPLE_COLLECTION and "birthyear" in props and date:
                deathyear_str = props.get("deathyear")
                birthyear_str = props.get("birthyear")
                if deathyear_str:
                    try:
                        deathyear = datetime(int(deathyear_str.split(".")[0]), 1,1)
                        if deathyear < date - timedelta(days=3650):  # 10 years before date
                            continue  # Skip old death date
                    except Exception:
                        print("failed")
                        pass  # If date parsing fails, don't filter out
                try:
                    birthyear = datetime(int(birthyear_str.split(".")[0]), 1,1)
                    if birthyear > date - timedelta(days=3650):  # 10 years before date
                        continue  # Skip old death date
                except Exception:
                    print("failed")
                    pass  # If date parsing fails, don't filter out
            # Check confidence/certainty score if available, include matches with high confidence
            if match.metadata.score > CERTAINTY:
                if collection_name == PEOPLE_COLLECTION:
                    results.append({
                        "name": props.get("name", name),
                        "description": props.get("description"),
                        "id": props.get("identifier"),
                        "url": props.get("url"),
                        "deathyear": props.get("deathyear"),
                        "score": match.metadata.score, 
                        "fuzzy_score": fuzzy_score
                    })
                elif collection_name == GEO_COLLECTION:
                    results.append({
                        "names": get_values_by_key_substring(props, "name"),
                        "type": props.get("type"),
                        "identifier": props.get("identifier"),
                        "url": props.get("url"),
                        "score": match.metadata.score, 
                        "fuzzy_score": fuzzy_score
                    })
                elif collection_name == ORG_COLLECTION:
                    names = props.get("name_de")+", "+props.get("name_fr")
                    results.append({
                        "names": names,
                        "url": props.get("url"),
                        "score": match.metadata.score, 
                        "fuzzy_score": fuzzy_score
                    })
    client.close()
    return results
def get_values_by_key_substring(dictionary, substring):
    return [value for key, value in dictionary.items() if substring in key]

def get_docinfo(doc_id):
    client.connect()
    collection = client.collections.use(DOC_COLLECTION)
    filters = Filter.by_property("doc_id").equal(doc_id)
    response = collection.query.fetch_objects(filters=filters, limit=1,include_vector=True)
    for obj in response.objects:
        file_name = obj.properties["file_name"]
        vector = obj.vector["default"]
    collection = client.collections.use(CHUNK_COLLECTION)
    filters = Filter.by_property("chunk_id").equal(doc_id)
    response = collection.query.fetch_objects(filters=filters, limit=1000)
    languages = set()
    chunks = []
    for obj in response.objects:
        languages.add(obj.properties["language"])
        chunks.append(obj.properties["chunk_id"])
    client.close()
    return file_name, vector, languages, chunks

def disambiguate(nlp, text, text_vector, text_date, client,  language, top_k=5):
    # Parse date string to datetime
    if isinstance(text_date, str):
        text_date = datetime.strptime(text_date, "%Y-%m-%d")
    entities = extract_entities(nlp,text)
    # Query Weaviate by entity type/collection
    people_results = query_weaviate(client, PEOPLE_COLLECTION, entities["PER"], text_vector, language, text_date, top_k)
    org_results = query_weaviate(client, ORG_COLLECTION, entities["ORG"], text_vector, language, None, top_k)
    place_results = query_weaviate(client, GEO_COLLECTION, entities["LOC"], text_vector, language, None, top_k )
    result = {
        "people": people_results,
        "organizations": org_results,
        "places": place_results
    }
    print(f"Got {len(result.get("people"))} people, {len(result.get("organizations"))} organizations and {len(result.get("places"))} places which match candidate entities")
    logging.info(f"Got {len(result.get("people"))} people, {len(result.get("organizations"))} organizations and {len(result.get("places"))} places which match candidate entities")
    return result

def ner_lang_chunks(chunks, language, vector, date, client):
    results = []
    if language == "de":
        nlp = spacy.load("de_core_news_lg")
    elif language == "fr":
        nlp = spacy.load("fr_core_news_lg")
    elif language == "it":
        nlp = spacy.load("it_core_news_lg")
    else:
        nlp = spacy.load("xx_ent_wiki_sm")
    for chunk in chunks:
        for order, text in chunk.items():
            print(f"extracting candidate entities from chunck {order}")
            logging.info(f"extracting candidate entities from chunck {order}")
            candidates = disambiguate(nlp,text,vector,date,client,"de")
            if candidates['people'] or candidates['organizations'] or candidates['places']:
                entities = get_roles_apertus(candidates,text,language)
                if entities:
                    results.append({'chunk_id': order, 'entities':entities})
                    print(f"LLM identified {len(entities.get("people"))} people, {len(entities.get("organizations"))} organizations and {len(entities.get("places"))} entities in chunk {order}")
                    logging.info(f"LLM identified {len(entities.get("people"))} people, {len(entities.get("organizations"))} organizations and {len(entities.get("places"))} entities in chunk {order}")
                else: 
                    print(f"LLM failed to confirm matched entities for chunk {order}")
                    logging.info(f"LLM failed to confirm matched entities for chunk {order}")
            else: 
                print(f"Matched no entities for chunk {order}")
                logging.info(f"Matched no entities for chunk {order}")
    return results

def definitive_id(doc_id, date):
    file_name , vector, languages, chunks = get_docinfo(doc_id)
    client.connect()
    collection = client.collections.use(CHUNK_COLLECTION)
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
            order = obj.properties["chunk_id"]
            if lang == "de":
                de_chunks.append({order:text})
            elif lang == "fr":
                fr_chunks.append({order:text})
            elif lang == "it":
                it_chunks.append({order:text})
            else:
                other_chunks.append({order:text})
    client.close
    if de_chunks:
        results.extend(ner_lang_chunks(de_chunks, "de", vector, date, client))
    if fr_chunks:
        results.extend(ner_lang_chunks(fr_chunks, "fr", vector, date, client))
    if it_chunks:
       results.extend(ner_lang_chunks(it_chunks, "it", vector, date, client))
    if other_chunks:
        results.extend(ner_lang_chunks(other_chunks, "other", vector, date, client))
    client.close
    return results


def store2graph(chunk_id, url,entity_type): 
    # Define the SPARQL endpoint URL for the repository
    endpoint_url = ENDPOINT_URL
    
    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper(endpoint_url)
    if SPARQL_USER and SPARQL_PASSWORD:
        sparql.setCredentials(SPARQL_USER, SPARQL_PASSWORD)
    sparql.setHTTPAuth(BASIC)  # Optional, add if authentication is required
    # sparql.setCredentials("username", "password")  # Uncomment if needed
    sparql.setMethod(POST)
    sparql.setRequestMethod(URLENCODED)

    #check url conformance
    # Define sample RDF triples in Turtle format to insert
    if url:
        if entity_type == "place":
            if "https://ld.admin.ch/" in url:
                insert_data = f"""
                PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
                INSERT DATA {{
                    GRAPH <{GRAPH}> {{
                <https://culture.ld.admin.ch/ais/{chunk_id}> rico:hasPlace <{url}> .

                }}
                }}
                """
            else: insert_data = None
        elif entity_type == "organization":
            if "https://culture.ld.admin.ch/sfa/" in url:
                insert_data = f"""
                PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
                INSERT DATA {{
                    GRAPH <{GRAPH}> {{
                <https://culture.ld.admin.ch/ais/{chunk_id}> rico:hasSubject <{url}> .

                }}
                }}
                """
            else: insert_data = None
        elif entity_type == "person": 
            if "http://hls-dhs-dss.ch/de/articles/" in url:
                insert_data = f"""
                PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
                INSERT DATA {{
                    GRAPH <{GRAPH}> {{
                <https://culture.ld.admin.ch/ais/{chunk_id}> rico:hasSubject <{url}> .

                }}
                }}
                """
            else: insert_data = None
        if insert_data:
            sparql.setQuery(insert_data)

        try:
            response = sparql.query()
            print(f"Stored entity links for chunk {chunk_id}")
            logging.info(f"Stored entity links for chunk {chunk_id}")
        except Exception as e:
            print(f"Failed to upload entity links for chunk {chunk_id}: {e}")
            logging.error(f"Failed to upload entity links for chunk {chunk_id}: {e}")

@app.post("/ner")
def orchestrator(input: TextInput):
    #iterate through collection of docs
    doc_id = input.doc_id
    date = input.date
    print(f"Performing named entity recognition on document {doc_id}")
    logging.info(f"Performing named entity recognition on document {doc_id}")
    try:
        doc_ents = definitive_id(doc_id, date)
        for chunk_ents in doc_ents:
            print(chunk_ents)
            entities = chunk_ents['entities']
            people = entities.get('people')
            organizations = entities.get('organizations')
            places = entities.get('places')
            if people:
                for person in people:
                    store2graph(chunk_ents['chunk_id'],person['url'],"person")
            if organizations:
                for organization in organizations:
                    store2graph(chunk_ents['chunk_id'],organization['url'],"organization")
            if places:
                for place in places:
                    store2graph(chunk_ents['chunk_id'],place['url'],"place")
    except Exception as e:
        logging.error(f"Error processing document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "running"}



