from SPARQLWrapper import SPARQLWrapper, TURTLE
from rdflib import Graph

# Define your endpoint and graph URI
endpoint_url = "https://ld.admin.ch/query"
graph_uri = "https://lindas.admin.ch/sfa/ais"

# Prepare the CONSTRUCT query with the GRAPH keyword
query = f"""
CONSTRUCT {{
  ?s ?p ?o
}}
WHERE {{
  GRAPH <{graph_uri}> {{
    ?s ?p ?o
  }}
}}
"""

# Set up SPARQLWrapper
sparql = SPARQLWrapper(endpoint_url)
sparql.setQuery(query)
sparql.setReturnFormat(TURTLE)

# Execute the query and get results in Turtle format
results = sparql.query().convert()

# Parse the results into an rdflib Graph
g = Graph()
g.parse(data=results, format="turtle")

# Serialize and save as TTL file
g.serialize(destination="ais_snapshot.ttl", format="turtle")
print("Graph exported successfully to ais_snapshot.ttl")

