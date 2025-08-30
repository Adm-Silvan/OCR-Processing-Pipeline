import os
import spacy
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Load spaCy model ---
nlp = spacy.load("de_core_news_lg")

# --- 2. Read .txt file ---
file_path = "c:/Users/Silvan/Documents/Obsidian Fusion/Uni/CAS Data Engineering/OCR-Processing-Pipeline/Entity_Extractor/1973-12-18.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# --- 3. Extract entities with spaCy ---
doc = nlp(text)
entities = []
for ent in doc.ents:
    entities.append({"text": ent.text, "label": ent.label_})

# --- 4. Prepare prompt for LLM disambiguation and relationship extraction ---
entity_summary = "\n".join([f"{e['text']} ({e['label']})" for e in entities])

print(entity_summary)

llm_prompt = f"""
Given the following extracted entities from a document:

{entity_summary}

And the document text:
\"\"\"
{text}
\"\"\"

1. For each entity, globally disambiguate it (e.g., link 'B u n d e s p r ä s i d e n t' to 'Bundespräsident' or 'Federal Councillor of Switzerland').
.').
2. Categorize all entities into: Organizations, Persons, Places, and Others.
3. For each Person, identify their role (e.g., Bundespräsident, Nationalrat, Geschäftsführer, Botschafter) and their relationships to Organizations and Places, if mentioned.
4. Return the result as a structured JSON object with this format:
{{
    "organizations": [{{"name": "...", "disambiguation": "..."}}, ...],
    "persons": [
        {{
            "name": "...",
            "disambiguation": "...",
            "role": "...",
            "organizations": ["..."],
            "places": ["..."]
        }},
        ...
    ],
    "places": [{{"name": "...", "disambiguation": "..."}}, ...],
    "others": [{{"name": "...", "disambiguation": "..."}}, ...]
}}
"""

# --- 5. Set up LangChain with Ollama ---
llm = OllamaLLM(model="mixtral:8x7b")  # Use your pulled model name

prompt = ChatPromptTemplate.from_template("{prompt}")
chain = prompt | llm

# --- 6. Run the LLM chain ---
result = chain.invoke({"prompt": llm_prompt})

# --- 7. Output the result ---
print(result)
