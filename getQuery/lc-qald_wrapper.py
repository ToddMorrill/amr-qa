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
#for e in data["questions"]:
for e in data:
	# print(e["query"]["sparql"])
	#statement = e["query"]["sparql"]
	statement = e["sparql_query"]
	sp.setQuery(statement)
	sp.setReturnFormat(JSON)
	r = sp.query().convert()
	r["id"] = e["_id"]
	result.append(r)


#out_file = open(sys.argv[2], "w")
#json.dump(result, out_file)
#out_file.close()
with open(sys.argv[2], "w") as outfile:
	json.dump(result, outfile, indent=2)

