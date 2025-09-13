from SPARQLWrapper import SPARQLWrapper, POST, TURTLE
import json

PATH = "C:/Users/Silvan/Data/OCR_Protocols/"
def load_iris(min_y, max_y):
    iris = []
    for year in range(min_y,max_y+1):
        for month in range(1,13):
            with open(PATH+f"Manifests/{year}-{month}.json", "r", encoding="utf-8") as file:
                data = json.load(file)
            for date, values in data.items():
                id = values[0]
                iris.append(id)
    return iris
                              

iris = load_iris(1950,1950)
 

# Split into batches if needed (SPARQL endpoints may have limits)

batch_size = 500  # Adjust as needed

endpoint = "https://lindas.admin.ch/query"

 

for i in range(0, len(iris), batch_size):
    batch = iris[i:i+batch_size]

    values_clause = "\n".join(f"<{iri}>" for iri in batch)

    query = f"""

    CONSTRUCT {{

    ?record ?p ?o .
    ?parent ?pp ?oo .
    }}
    WHERE {{
    VALUES ?record {{

    {values_clause}

    }}

    ?record ?p ?o .

    OPTIONAL {{

    ?record https://www.ica.org/standards/RiC/ontology#isOrWasIncludedIn+ ?parent .

    ?parent ?pp ?oo .

    }}

    }}

    """

    sparql = SPARQLWrapper(endpoint)

    sparql.setQuery(query)

    sparql.setMethod(POST)

    sparql.setReturnFormat(TURTLE)

    results = sparql.query().convert()
    with open(f"ais_subgraph_{i//batch_size}.ttl", "wb") as out:
        out.write(results)