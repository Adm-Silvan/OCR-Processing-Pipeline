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
    
    Beispielsweise bei folgenden Entitäten:
    {{"people":[{{'birthyear': '1851', 'deathyear': '1924.0', 'description': '∗︎\xa02.9.1851 St. Gallen, ✝︎\xa026.11.1924 St. Gallen, reformiert, von St. Gallen. Sohn des Karl Jakob. ∞︎\xa0Anna Pauline Ze  ellweger. Kantonsschule St. Gallen, danach Handelslehre. Leitende Position im St. Galler Textilhandelshaus Ulrich von Caspar Vonwiller. Aus diesem wurde 1892 die weltweit tätige Stickerei-Exportfirma Hoffmann, Huber & Co. (ab 1909 Stickerei-AG Union), die Max Hoffmann als Miteigner leitete. Hoffmann erlebte den grossen Aufschwung der Maschinenstickerei, deren Anliegen er mit rastlosem Engagement förderte, unter anderem als Delegierter in Handelsvertragsverhandlungen und 1902-1917 als Mitglied des Kaufmännischen Direktoriums St. Gallen. 1877 Mitgründer und dann Förderer des Konzertvereins St. Gallen.', 'name': 'Max Hoffmann',  'url': 'http://hls-dhs-dss.ch/de/articles/027877/2007-12-18/'}}, {{'roles': None, 'deathyear': '1927.0', 'birthyear': '1857', 'name': 'Arthur Hoffmann', 'description': '∗︎\xa019.6.1857 (nicht der 18..6.) St. Gallen, ✝︎\xa023.7.1927 St. Gallen, reformiert, von St. Gallen. Jurist, Ständerat des Kantons St. Gallen, freisinniger Bundesrat.', 'url': 'http://hls-dhs-dss.ch/de/artticles/003991/2022-03-14/'}}]}}
    und den folgenden Text:
    "Ausserordentliche 118. Sitzung des Schweizerischen Bundesrates. Montag, den 19.Oktober 1914, vormittags 9 Uhr. Präsidium : Herr Bundespräsident Hoffmann. Mitglieder: Herr Vizepräsident Motta und Herren Bundesräte Müller, Decoppet und Calonder. Abwesend (wegen Unpässlichkeit): Herren Bundes- räte Forrer und Schulthess. Aktuariat : Herr Bundeskanzler Schatzmann und Herr Vize- kanzler David. Départemental"
    Nur die zweite Entität hat die richtige Rolle „Bundesrat“ in ihrer Beschreibung, die mit „Bundespräsident“ im Text übereinstimmt. Es sollte nur die richtige Entität zurückgegeben werden. Die Ausgabe sollte lauten:
    {{"people": [{{"name": "Arthur Hoffmann", "url": "http://hls-dhs-dss.ch/de/artticles/003991/2022-03-14/",}} ]}}
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
    
    Par exemple, étant donné les entités suivantes :
    {{"people":[{{'birthyear': '1851', 'deathyear': '1924.0', 'description': '∗︎\xa02.9.1851 St. Gallen, ✝︎\xa026.11.1924 St. Gallen, reformiert, von St. Gallen. Sohn des Karl Jakob. ∞︎\xa0Anna Pauline Ze  ellweger. Kantonsschule St. Gallen, danach Handelslehre. Leitende Position im St. Galler Textilhandelshaus Ulrich von Caspar Vonwiller. Aus diesem wurde 1892 die weltweit tätige Stickerei-Exportfirma Hoffmann, Huber & Co. (ab 1909 Stickerei-AG Union), die Max Hoffmann als Miteigner leitete. Hoffmann erlebte den grossen Aufschwung der Maschinenstickerei, deren Anliegen er mit rastlosem Engagement förderte, unter anderem als Delegierter in Handelsvertragsverhandlungen und 1902-1917 als Mitglied des Kaufmännischen Direktoriums St. Gallen. 1877 Mitgründer und dann Förderer des Konzertvereins St. Gallen.', 'name': 'Max Hoffmann',  'url': 'http://hls-dhs-dss.ch/de/articles/027877/2007-12-18/'}}, {{'roles': None, 'deathyear': '1927.0', 'birthyear': '1857', 'name': 'Arthur Hoffmann', 'description': '∗︎\xa019.6.1857 (nicht der 18..6.) St. Gallen, ✝︎\xa023.7.1927 St. Gallen, reformiert, von St. Gallen. Jurist, Ständerat des Kantons St. Gallen, freisinniger Bundesrat.', 'url': 'http://hls-dhs-dss.ch/de/artticles/003991/2022-03-14/'}}]}}
    et le texte suivant :
    "Ausserordentliche 118. Sitzung des Schweizerischen Bundesrates. Montag, den 19.Oktober 1914, vormittags 9 Uhr. Präsidium : Herr Bundespräsident Hoffmann. Mitglieder: Herr Vizepräsident Motta und Herren Bundesräte Müller, Decoppet und Calonder. Abwesend (wegen Unpässlichkeit): Herren Bundes- räte Forrer und Schulthess. Aktuariat : Herr Bundeskanzler Schatzmann und Herr Vize- kanzler David. Départemental"
    Seule la deuxième entité a le rôle correct « Bundesrat » dans sa description, correspondant à Bundespräsident dans le texte. Seule l'entité correcte doit être renvoyée. Le résultat doit être :
    {{"people": [{{"name": "Arthur Hoffmann", "url": "http://hls-dhs-dss.ch/de/artticles/003991/2022-03-14/",}} ]}}
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
    
    Ad esempio, date le seguenti entità:
    {{"people":[{{'birthyear': '1851', 'deathyear': '1924.0', 'description': '∗︎\xa02.9.1851 St. Gallen, ✝︎\xa026.11.1924 St. Gallen, reformiert, von St. Gallen. Sohn des Karl Jakob. ∞︎\xa0Anna Pauline Ze  ellweger. Kantonsschule St. Gallen, danach Handelslehre. Leitende Position im St. Galler Textilhandelshaus Ulrich von Caspar Vonwiller. Aus diesem wurde 1892 die weltweit tätige Stickerei-Exportfirma Hoffmann, Huber & Co. (ab 1909 Stickerei-AG Union), die Max Hoffmann als Miteigner leitete. Hoffmann erlebte den grossen Aufschwung der Maschinenstickerei, deren Anliegen er mit rastlosem Engagement förderte, unter anderem als Delegierter in Handelsvertragsverhandlungen und 1902-1917 als Mitglied des Kaufmännischen Direktoriums St. Gallen. 1877 Mitgründer und dann Förderer des Konzertvereins St. Gallen.', 'name': 'Max Hoffmann',  'url': 'http://hls-dhs-dss.ch/de/articles/027877/2007-12-18/'}}, {{'roles': None, 'deathyear': '1927.0', 'birthyear': '1857', 'name': 'Arthur Hoffmann', 'description': '∗︎\xa019.6.1857 (nicht der 18..6.) St. Gallen, ✝︎\xa023.7.1927 St. Gallen, reformiert, von St. Gallen. Jurist, Ständerat des Kantons St. Gallen, freisinniger Bundesrat.', 'url': 'http://hls-dhs-dss.ch/de/artticles/003991/2022-03-14/'}}]}}
    e il seguente testo:
    "Ausserordentliche 118. Sitzung des Schweizerischen Bundesrates. Montag, den 19.Oktober 1914, vormittags 9 Uhr. Präsidium : Herr Bundespräsident Hoffmann. Mitglieder: Herr Vizepräsident Motta und Herren Bundesräte Müller, Decoppet und Calonder. Abwesend (wegen Unpässlichkeit): Herren Bundes- räte Forrer und Schulthess. Aktuariat : Herr Bundeskanzler Schatzmann und Herr Vize- kanzler David. Départemental"
    Solo la seconda entità ha il ruolo corretto “Bundesrat” nella sua descrizione, corrispondente a Bundespräsident nel testo. Dovrebbe essere restituita solo l'entità corretta. Il risultato dovrebbe essere:
    {{"people": [{{"name": "Arthur Hoffmann", "url": "http://hls-dhs-dss.ch/de/artticles/003991/2022-03-14/",}} ]}}
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
