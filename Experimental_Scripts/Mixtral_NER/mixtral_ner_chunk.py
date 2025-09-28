import spacy
from datetime import datetime, timedelta
import weaviate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from weaviate.classes.query import Filter
from weaviate.connect import ConnectionParams
from thefuzz import fuzz
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import ast
import re
import tracemalloc
from SPARQLWrapper import SPARQLWrapper, POST, BASIC, URLENCODED
from codecarbon import EmissionsTracker
from mixtral_get_roles import  get_roles
tracemalloc.start()


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
    entities = {"PER": set(), "ORG": set(), "GPE": set()}
    for ent in doc.ents:
        if ent.label_ in entities:
            # Normalize entity text by stripping whitespace
            entities[ent.label_].add(ent.text.strip())
    entities["PER"] = group_and_reduce_entities(list(entities["PER"]), threshold=85)
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
        query_args = {
            "query": name,
            "vector": text_vector,
            "alpha": 0.25,
            "limit": top_k,
            "return_metadata": weaviate.classes.query.MetadataQuery(score=True, explain_score=True),
            "return_properties": ["url"]
        }
        if collection_name == "People":
            query_args["return_properties"].extend(["name", "description", "hls_id", "deathyear"])
        if collection_name == "Place":
            query_args["return_properties"].extend(["name_de", "name_fr", "name_it", "type", "identifier", ])
        if collection_name == "Organization":
            query_args["return_properties"].extend(["name_de", "name_fr"])

        response = collection.query.hybrid(**query_args)
        for match in response.objects:
            props = match.properties
            if collection_name == "People":
                candidate_name = props.get("name", "")
            elif collection_name == "Organization":
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
                if collection_name == "People":
                    results.append({
                        "name": props.get("name", name),
                        "description": props.get("description"),
                        "id": props.get("hls_id"),
                        "url": props.get("url"),
                        "deathyear": props.get("deathyear"),
                        "score": match.metadata.score, 
                        "fuzzy_score": fuzzy_score
                    })
                elif collection_name == "Place":
                    results.append({
                        "names": get_values_by_key_substring(props, "name"),
                        "type": props.get("type"),
                        "identifier": props.get("identifier"),
                        "url": props.get("url"),
                        "score": match.metadata.score, 
                        "fuzzy_score": fuzzy_score
                    })
                elif collection_name == "Organization":
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

def disambiguate(nlp, text, text_vector, text_date, client,  language, top_k=5):
    # Parse date string to datetime
    if isinstance(text_date, str):
        text_date = datetime.strptime(text_date, "%Y-%m-%d")
    entities = extract_entities(nlp,text)
    # Query Weaviate by entity type/collection
    people_results = query_weaviate(client, "People", entities["PER"], text_vector, language, text_date, top_k)
    org_results = query_weaviate(client, "Organization", entities["ORG"], text_vector, language, None, top_k)
    place_results = query_weaviate(client, "Place", entities["GPE"], text_vector, language, None, top_k )

    return {
        "people": people_results,
        "organizations": org_results,
        "places": place_results
    }
    

def definitive_id(file,date):
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
        nlp = spacy.load("de_core_news_lg")
        for chunk in de_chunks:
            for order, text in chunk.items():
                candidates = disambiguate(nlp,text,vector,date,client,"de")
                if candidates['people'] or candidates['organizations'] or candidates['places']:
                    people = get_roles(candidates,text,"de")
                    if people:
                        results.append({'chunk_id': order, 'people':people})
                        print(f"Extracted entities from chunk {order}")
                        print(people)
                    else: 
                        print(f"LLM failed to confirm matched entities for chunk {order}")
                        print()
                else: print(f"Matched no entities for chunk {order}")
    if fr_chunks:
        nlp = spacy.load("fr_core_news_lg")
        for chunk in fr_chunks:
            for order, text in chunk.items():
                candidates = disambiguate(nlp,text,vector,date,client,"fr")
                if candidates['people']:
                    people = get_roles(candidates,text,"fr")
                    if people:
                        results.append({'chunk_id': order, 'people':people})
                        print(f"Extracted entities from chunk {order}")
                    else: print(f"LLM failed to confirm matched entities for chunk {order}")
                else: print(f"Matched no entities for chunk {order}")
    if it_chunks:
        nlp = spacy.load("it_core_news_lg")
        for chunk in it_chunks:
            for order, text in chunk.items():
                candidates = disambiguate(nlp,text,vector,date,client,"it")
                if candidates['people']:
                    people = get_roles(candidates,text,"it")
                    if people:
                        results.append({'chunk_id': order, 'people':people})
                        print(f"Extracted entities from chunk {order}")
                    else: print(f"LLM failed to confirm matched entities for chunk {order}")
                else: print(f"Matched no entities for chunk {order}")
    if other_chunks:
        nlp = spacy.load("xx_ent_wiki_sm")
        for chunk in other_chunks:
            for order, text in chunk.items():
                candidates = disambiguate(nlp,text,vector,date,client,"de")
                if candidates['people']:
                    people = get_roles(candidates,text,"de")
                    if people:
                        results.append({'chunk_id': order, 'people':people})
                        print(f"Extracted entities from chunk {order}")
                    else: print(f"LLM failed to confirm matched entities for chunk {order}")
                else: print(f"Matched no entities for chunk {order}")
    client.close
    return results
    
def get_all_files(base_path):
    file_paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(root, file)
            full_path = os.path.normpath(full_path).replace(os.sep, '/')
            file_paths.append(full_path)
    return file_paths

def clean_llm_output(text):
    # Replace typographic quotes with ASCII quotes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    
    # Replace single quotes around keys and values with double quotes for JSON compatibility
    # Optional: only if parsing with json.loads()
    # text = re.sub(r"(?<=[:,\s])'(.*?)'(?=[,\s}])", r'"\1"', text)

    # Escape newlines and tabs inside strings to avoid parse errors    
    # Remove trailing commas before closing braces or brackets (common JSON mistake)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    
    # Remove zero-width spaces or other invisible characters
    text = text.replace("\u200b", "").replace("\ufeff", "")
    
    # Optionally, remove non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\r', '\t'])
    
    return text

def store2graph(chunk_id, url,entity_type): 
    # Define the SPARQL endpoint URL for the repository
    endpoint_url = ENDPOINT_URL

    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper(endpoint_url)
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
        except Exception as e:
            print(f"[ERROR] Failed to upload entity links for chunk {chunk_id}: {e}")

def orchestrator(file,date):
    doc_ents = definitive_id(file,date)
    for chunk_ents in doc_ents:
        print(chunk_ents)
        entities = chunk_ents['people']
        people = entities['people']
        organizations = entities['organizations']
        places = entities['places']
        for person in people:
            store2graph(chunk_ents['chunk_id'],person['url'],"person")
        for organization in organizations:
            store2graph(chunk_ents['chunk_id'],organization['url'],"organization")
        for place in places:
            store2graph(chunk_ents['chunk_id'],place['url'],"place")
        


ENDPOINT_URL = "http://localhost:7200/repositories/AIS/statements"
GRAPH = "https://lindas.admin.ch/sfa/ais/entities/apertus"
CERTAINTY = 0.7
FUZZ = 75

if __name__ == "__main__":
    """    file_name = "1958-01-25.txt"
        basename = file_name.rsplit('.', 1)[0]
        year, month, _ = basename.split('-')
        file = f"C:/Users/Silvan/Data/OCR_Protocols/{year}/{month}/{file_name}"
        date = file.split("/")[-1][:-4]
        print(date)
        with EmissionsTracker(save_to_file=True, output_dir="C:/Users/Silvan/Data/Logs/NER", output_file="emissions.csv") as tracker:
            print(orchestrator(file,date))
    """
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
    for obj in collection.iterator(return_properties="file_name"):
        file_name = obj.properties["file_name"]
        basename = file_name.rsplit('.', 1)[0]
        year, month, _ = basename.split('-')
        file = f"C:/Users/Silvan/Data/OCR_Protocols/{year}/{month}/{file_name}"
        date = file.split("/")[-1][:-4]
        print(date)
        with EmissionsTracker(save_to_file=True, output_dir="C:/Users/Silvan/Data/Logs/NER", output_file="emissions.csv") as tracker:
            print(orchestrator(file,date))  