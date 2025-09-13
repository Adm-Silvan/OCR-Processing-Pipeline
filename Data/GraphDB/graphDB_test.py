from SPARQLWrapper import SPARQLWrapper, POST, BASIC, URLENCODED

# Define the SPARQL endpoint URL for the repository
endpoint_url = "http://localhost:7200/repositories/AIS/statements"

# Initialize SPARQLWrapper
sparql = SPARQLWrapper(endpoint_url)
sparql.setHTTPAuth(BASIC)  # Optional, add if authentication is required
# sparql.setCredentials("username", "password")  # Uncomment if needed
sparql.setMethod(POST)
sparql.setRequestMethod(URLENCODED)

# Define sample RDF triples in Turtle format to insert
insert_data = """
PREFIX ex: <http://example.org/>
INSERT DATA {
    GRAPH <http://example.org/graph/test> {
  ex:subject1 ex:predicate1 "object1" .
  ex:subject2 ex:predicate2 "object2" .
}
}
"""

sparql.setQuery(insert_data)

try:
    response = sparql.query()
    print("Data uploaded successfully.")
except Exception as e:
    print("Failed to upload data:", e)
