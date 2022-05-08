from SPARQLWrapper import SPARQLWrapper, JSON
import json
import sys


#read data file
file_name = sys.argv[1]
with open(file_name, 'r') as f:
	data = json.load(f)

empty = 0
for e in data:
	try:
		e1 = e["results"]["bindings"]
		if len(e1) == 0:
			empty += 1
	except:
		# true/false type question
		e2 = e["boolean"]

print(empty)
percent = empty/(len(data))*100
print("%f of sparql query return empty result."%(percent) )