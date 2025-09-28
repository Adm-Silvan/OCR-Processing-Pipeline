def get_roles(combined_entities,text,language):
    de_llm_prompt = f"""
    Angenommen, die folgenden Entitäten wurden aus einem Dokument extrahiert:

    {combined_entities}

    Und der Dokumenttext lautet:
    \"\"\"
    {text}
    \"\"\"


    1. Identifizieren Sie die Rolle jeder Person (z. B. Bundespräsident, Nationalrat, Generaldirektor, Botschafter).
    2. Überprüfen Sie, ob die Rolle im Text mit der Rolle in ihrer Beschreibung übereinstimmt.
    3. Bei Personen, bei denen aufgrund übereinstimmender Rollen, eines übereinstimmenden Todesjahres und einer hohen Übereinstimmung eine Übereinstimmung wahrscheinlich erscheint, senden Sie das Wörterbuch mit ihren Informationen und ihren Rollen zurück.
    4. Stellen Sie fest, ob die Personen und Organisationen aus dem Eintrag im Text vorkommen, und fügen Sie sie gegebenenfalls zum Dictionary hinzu. 
    {{
        "people": [
            {{
                "name": "...",
                "description": "... ",
                "id": "... ",
                "url": "... ",
                "deathyear": "... ",
                "score": "... ",
                "id": "... ",
                "roles": ["...", "...", ...]
            }},
            ...
        ],
        "places": [{{
                'name_de': "...",
                'name_fr': "...",
                'name_it': "...",
                'identifier': "...",
                'url': "..."
                }}, 
                ...
            ],
        "organizations": [{{
                'name_de': "...",
                'name_fr': "...",
                'url': "..."
                }}, 
                ...
            ],
    }}
    Geben Sie nur genau diesen Dictionary zurück und sonst nichts und stellen Sie sicher, dass es in der richtigen Python-Syntax wiedergegeben wird.
    """
    fr_llm_prompt = f"""Étant donné les entités extraites suivantes d'un document :

    {combined_entities}

    Et le texte du document :
    \"\"\"
    {text}
    \"\"\"


    1. Identifiez le rôle de chaque personne (par exemple, président fédéral, conseiller national, directeur général, ambassadeur).
    2. Vérifiez si le rôle dans le texte correspond au rôle utilisé dans leur description.
    3. Pour les personnes pour lesquelles une correspondance semble probable en raison de rôles correspondants, d'une année de décès concordante et d'une correspondance étroite, renvoyez le dictionnaire contenant leurs informations, ainsi que leurs rôles.
    4. Identifiez si les personnes et les organisations issues de l'entrée sont présentes dans le texte et, si tel est le cas, ajoutez-les au dictionnaire.
    {{
        "people": [
            {{
                "name": "...",
                "description": "... ",
                "id": "... ",
                "url": "... ",
                "deathyear": "... ",
                "score": "... ",
                "id": "... ",
                "roles": ["...", "...", ...]
            }},
            ...
        ],
        "places": [{{
                'name_de': "...",
                'name_fr': "...",
                'name_it': "...",
                'identifier': "...",
                'url': "..."
                }}, 
                ...
            ],
        "organizations": [{{
                'name_de': "...",
                'name_fr': "...",
                'url': "..."
                }}, 
                ...
            ],
    }}
    Renvoie uniquement ce dictionnaire et rien d'autre en veillant à ce qu'il soit dans une syntaxe Python correcte.
    """
    it_llm_prompt = f"""Date le seguenti entità estratte da un documento:

    {combined_entities}

    E il testo del documento:
    \"\"\"
    {text}
    \"\"\"


    1. Identificate il ruolo di ciascuna persona (ad esempio, presidente federale, consigliere nazionale, direttore generale, ambasciatore).
    2. Verificate se il ruolo nel testo corrisponde al ruolo utilizzato nella loro descrizione.
    3. Per le persone per le quali sembra probabile una corrispondenza a causa di ruoli corrispondenti, un anno di morte concordante e una stretta corrispondenza, rinvia il dizionario contenente le loro informazioni, insieme ai loro ruoli.
    4. Identifica se le persone e le organizzazioni inserite sono presenti nel testo e, in caso affermativo, aggiungile al dizionario.
    {{
        "people": [
            {{
                "name": "...",
                "description": "... ",
                "id": "... ",
                "url": "... ",
                "deathyear": "... ",
                "score": "... ",
                "id": "... ",
                "roles": ["...", "...", ...]
            }},
            ...
        ],
        "places": [{{
                'name_de': "...",
                'name_fr': "...",
                'name_it': "...",
                'identifier': "...",
                'url': "..."
                }}, 
                ...
            ],
        "organizations": [{{
                'name_de': "...",
                'name_fr': "...",
                'url': "..."
                }}, 
                ...
            ],
    }}
    Restituisci solo questo dizionario e nient'altro, assicurandoti che sia scritto nella sintassi Python corretta.
    """
    # --- 5. Set up LangChain with Ollama ---
    llm = OllamaLLM(model="mixtral:8x7b")  # Use your pulled model name

    prompt = ChatPromptTemplate.from_template("{prompt}")
    chain = prompt | llm

    # --- 6. Run the LLM chain ---
    if language == "de":
        result = chain.invoke({"prompt": de_llm_prompt})
    elif language == "fr":
        result = chain.invoke({"prompt": fr_llm_prompt})
    elif language == "it":
        result = chain.invoke({"prompt": it_llm_prompt})
    else: 
        result = chain.invoke({"prompt": de_llm_prompt})
    match = re.search(r'(\{.*\})', result, re.DOTALL)
    if match:
        dict_str = match.group(1)
        cleaned_dict_str = clean_llm_output(dict_str)
        try:
            result_dict = ast.literal_eval(cleaned_dict_str)
            for result in result_dict['people']:
                if not result['url']:
                    result_dict['people'].remove(result)
            for result in result_dict['places']:
                if not result['url']:
                    result_dict['places'].remove(result)
            for result in result_dict['organizations']:
                if not result['url']:
                    result_dict['organizations'].remove(result)
            return result_dict
        except Exception as e:
            print(f"Failed to parse dictionary string: {e}")
            print(cleaned_dict_str)
            result_dict = None
    else:
        print("No dictionary found in text")