from SPARQLWrapper import SPARQLWrapper, POST, TURTLE

endpoint = "https://lindas.admin.ch/query"
query = """
CONSTRUCT {
?country ?p ?o .
}
WHERE {
GRAPH <https://lindas.admin.ch/territorial> {
?country a <http://schema.org/Country> ;
?p ?o .
}
}


"""
sparql = SPARQLWrapper(endpoint)
sparql.setQuery(query)
sparql.setMethod(POST)
sparql.setReturnFormat(TURTLE)
results = sparql.query().convert()
with open("countries.ttl", "wb") as out:
    out.write(results)

endpoint = "https://lindas.admin.ch/query"
query = """
CONSTRUCT {
?commune ?p ?o .
}
WHERE {
GRAPH <https://lindas.admin.ch/fso/register> {
?commune a <https://schema.ld.admin.ch/Municipality> ;
?p ?o .
}
}
"""
sparql = SPARQLWrapper(endpoint)
sparql.setQuery(query)
sparql.setMethod(POST)
sparql.setReturnFormat(TURTLE)
results = sparql.query().convert()
with open("municipalities.ttl", "wb") as out:
    out.write(results)

endpoint = "https://lindas.admin.ch/query"
query = """
CONSTRUCT {
?canton ?p ?o .
}
WHERE {
GRAPH <https://lindas.admin.ch/fso/register> {
?canton a <https://schema.ld.admin.ch/Canton> ;
?p ?o .
}
}
"""
sparql = SPARQLWrapper(endpoint)
sparql.setQuery(query)
sparql.setMethod(POST)
sparql.setReturnFormat(TURTLE)
results = sparql.query().convert()
with open("cantons.ttl", "wb") as out:
    out.write(results)