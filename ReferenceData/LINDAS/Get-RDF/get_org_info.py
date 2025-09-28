from SPARQLWrapper import SPARQLWrapper, POST, TURTLE

endpoint = "https://lindas.admin.ch/query"
query = """
CONSTRUCT {
# Cube metadata
<https://culture.ld.admin.ch/sfa/StateAccounts_Office/9> ?cubeP ?cubeO .
# Observation set and its metadata
?obsSet ?obsSetP ?obsSetO .
# Observations and their metadata
?observation ?obsP ?obsO .
# 1st level: objects linked from observation
?obsO ?o1P ?o1O .
# 2nd level: objects linked from the first level
?o1O ?o2P ?o2O .
}
WHERE {
GRAPH <https://lindas.admin.ch/sfa/cube> {
# Cube metadata
<https://culture.ld.admin.ch/sfa/StateAccounts_Office/9> ?cubeP ?cubeO .
# Observation set
<https://culture.ld.admin.ch/sfa/StateAccounts_Office/9> <https://cube.link/observationSet> ?obsSet .
?obsSet ?obsSetP ?obsSetO .
# Observations in the set
?obsSet <https://cube.link/observation> ?observation .
?observation ?obsP ?obsO .
# 1st level: all triples about objects linked from observation
OPTIONAL { ?obsO ?o1P ?o1O . }
# 2nd level: all triples about objects linked from the first level
OPTIONAL { ?o1O ?o2P ?o2O . }
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