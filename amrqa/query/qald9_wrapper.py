from SPARQLWrapper import SPARQLWrapper, JSON
import json
import sys


endpoint = 'https://dbpedia.org/sparql'
sp = SPARQLWrapper(endpoint)

#read data file
file_name = sys.argv[1]
with open(file_name, 'r') as f:
	data = json.load(f)

result = []
i = 0
for e in data["questions"]:
	# print(e["query"]["sparql"])
	i += 1
	if i < 50:
		statement = e["query"]["sparql"]
		sp.setQuery(statement)
		sp.setReturnFormat(JSON)
		r = sp.query().convert()
		result.append(r)



out_file = open(sys.argv[2], "w")
json.dump(result, out_file)
out_file.close()

