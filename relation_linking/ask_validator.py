from SPARQLWrapper import SPARQLWrapper, XML

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery("""
    PREFIX res: <http://dbpedia.org/resource/>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    ASK WHERE {
        res:Abraham_Lincoln dbo:deathPlace ?a.
    }
""")
sparql.setReturnFormat(XML)
results = sparql.query().convert()
print(results.toxml())

# ?a dbo:deathPlace res:Abraham_Lincoln.