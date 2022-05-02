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
for k,e in data.items():
	# print(e["query"]["sparql"])
	try:
		statement = e["sparql"]
		sp.setQuery(statement)
		sp.setReturnFormat(JSON)
		r = sp.query().convert()
		r["status"] = "success"
	except:
		r = {}
		r["status"] = "fail"
	r["id"] = k
	result.append(r)




out_file = open(sys.argv[2], "w")
json.dump(result, out_file)
out_file.close()

