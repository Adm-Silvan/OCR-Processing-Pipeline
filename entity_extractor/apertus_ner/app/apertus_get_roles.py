from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
import re
import requests
import json
import os


global api_url
api_url = os.environ.get("LLM_ENPOINT", "http://vllm-server:8000/v1/chat/completions")


def extract_balanced_braces(text):
    start = None
    brace_count = 0
    for i, char in enumerate(text):
        if char == '{':
            if start is None:
                start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start is not None:
                return text[start:i+1]
    return None  # no balanced braces found


def call_vllm_chat_completion(prompt, model="swiss-ai/Apertus-8B-Instruct-2509", api_url="http://vllm-server:8000/v1/chat/completions"):
    headers = {
        "Content-Type": "application/json",
        # Include Authorization if needed
        #"Authorization": "Bearer YOUR_TOKEN"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a plausibility checking system for named entity recognition that only returns json."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 3000,
        "temperature": 0.7,
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    response_json = response.json()
    result_text = response_json["choices"][0]["message"]["content"]
    return result_text

def clean_llm_output(text):
    # Fix inconsistent quotes
    text = text.strip()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Remove trailing commas before } or ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # Replace non-breaking space and remove zero-width chars
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    # Keep only printable or newline/tab chars
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\r', '\t'])
    # Replace Python-style literals with JSON literals only if safe
    text = text.replace("None", "null").replace("True", "true").replace("False", "false")
    text = str(text)
    try:
        data = json.loads(text)
    except Exception as e:
        print("Failed to parse JSON:", e)
        print(text)
        data = None
    return data

def get_roles_apertus(combined_entities, text, language):
    # Prompt selection as in your template
    de_llm_prompt = f"""
    Angenommen, die folgenden Entitäten wurden aus einem Dokument extrahiert:

    {combined_entities}

    Und der Dokumenttext lautet:
    \"\"\"
    {text}
    \"\"\"

    1. Identifizieren Sie für jede Person in den Entitäten deren Rolle anhand des Dokumenttextes.
    2. Überprüfen Sie, ob die von Ihnen identifizierten Rollen mit den Rollen übereinstimmen, die Sie in der Beschreibung identifizieren können.
    3. Für die Personen deren Rollen im Text und in der Beschreibung übereinstimmen, fügen Sie deren Name und URL zum folgenden Wörterbuchstruktur hinzu. 
    {{"people": [{{"name": "...", "url": "... ",}}, ... ]}}
    Geben Sie nur genau diesen Dictionary zurück und sonst nichts und stellen Sie sicher, dass es in der richtigen JSON-Syntax wiedergegeben wird.
    """
    fr_llm_prompt = f"""Étant donné les entités extraites suivantes d'un document :

    {combined_entities}

    Et le texte du document :
    \"\"\"
    {text}
    \"\"\"

    1. Pour chaque personne dans les entités, identifiez son rôle en vous basant sur le texte du document.
    2. Vérifiez si les rôles que vous avez identifiés correspondent à ceux que vous pouvez identifier dans la description.
    3. Pour les personnes dont les rôles correspondent dans le texte et dans la description, ajoutez leur nom et leur URL à la structure du dictionnaire suivante. 
    Renvoie uniquement ce dictionnaire et rien d'autre en veillant à ce qu'il soit dans une syntaxe JSON correcte.
    """
    it_llm_prompt = f"""Date le seguenti entità estratte da un documento:

    {combined_entities}

    E il testo del documento:
    \"\"\"
    {text}
    \"\"\"

    1. Per ogni persona presente nelle entità, identificare il proprio ruolo in base al testo del documento.
    2. Verificare se i ruoli identificati corrispondono a quelli identificabili nella description.
    3. Per le persone i cui ruoli corrispondono nel testo e nella descrizione, aggiungi il loro nome e URL alla seguente struttura del dizionario. 
    {{"people": [{{"name": "...", "url": "... ",}}, ... ]}}
    Restituisci solo questo dizionario e nient'altro, assicurandoti che sia scritto nella sintassi JSON corretta.
    """
    if language == "de":
        prompt = de_llm_prompt
    elif language == "fr":
        prompt =   fr_llm_prompt# your existing fr prompt
    elif language == "it":
        prompt =   it_llm_prompt# your existing it prompt
    else:
        prompt =   de_llm_prompt# fallback
    attempts = 0
    while True:    
        result = call_vllm_chat_completion(prompt, api_url)
        # regex search and parse dictionary as before
        match = extract_balanced_braces(result)
        if match:
            result_dict = clean_llm_output(match)
            if result_dict:
                try: 
                    for result in result_dict.get('people'):
                        if not result.get('url'):
                            result_dict['people'].remove(result)
                    return result_dict
                except:
                    print("Failed to parse json as python dict")
                    if attempts > 2: 
                        print(f"Failed to parse dictionary {attempts} times. Last result:\noutput: {match}\npost cleaning: {result_dict}")
                        return None
                    else:
                        print("Retrying...")
                        attempts += 1
            else:
                print(f"Failed to parse json")
                if attempts > 2: 
                    print(f"Failed to parse dictionary {attempts} times. Last result:\noutput: {match}\n{result_dict}")
                    return None
                else:
                    print("Retrying...")
                    attempts += 1
        else:
            print("No dictionary found in text")
            print(result)
            return None
