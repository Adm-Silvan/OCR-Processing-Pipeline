from SPARQLWrapper import SPARQLWrapper, POST, TURTLE
from rdflib import Graph
import time
import random

# Define your endpoint and graph URI
endpoint_url = "https://ld.admin.ch/query"
graph_uri = "https://lindas.admin.ch/sfa/ais"
chunk_size = 100000
offset = 80880000
batch = 10000

while True:
  query = f"""
  CONSTRUCT {{
  ?s ?p ?o .
  }}
  WHERE {{
  GRAPH <{graph_uri}> {{
  ?s ?p ?o .
  }}
  }}
  LIMIT {chunk_size}
  OFFSET {offset}
  """
  sparql = SPARQLWrapper(endpoint_url)
  sparql.setQuery(query)
  sparql.setMethod(POST)
  sparql.setReturnFormat(TURTLE)
  results = sparql.query().convert()
  if not results or results.strip() == b'':
    break  # No more data
  with open(f"ais_export_part_{batch}.ttl", "wb") as out:
    out.write(results)
    print(f"Exported batch {batch} (offset {offset})")
    offset += chunk_size
    batch += 1
  sleepr = random.choice(range(1,10))
  print(sleepr)
  time.sleep(sleepr)