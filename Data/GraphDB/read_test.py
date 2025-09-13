from SPARQLWrapper import SPARQLWrapper, JSON

# SPARQL endpoint URL for querying
endpoint_url = "http://localhost:7200/repositories/AIS"

# Initialize SPARQLWrapper
sparql = SPARQLWrapper(endpoint_url)

# Define SPARQL SELECT query to retrieve 10 triples
query = """
SELECT ?s ?p ?o
WHERE {
GRAPH <http://example.org/graph/test> {
  ?s ?p ?o .
}
}
LIMIT 1000
"""

sparql.setQuery(query)
sparql.setReturnFormat(JSON)

try:
    results = sparql.query().convert()
    # Print the retrieved triples
    for result in results["results"]["bindings"]:
        s = result["s"]["value"]
        p = result["p"]["value"]
        o = result["o"]["value"]
        print(f"Subject: {s}, Predicate: {p}, Object: {o}")
except Exception as e:
    print("Query failed:", e)
