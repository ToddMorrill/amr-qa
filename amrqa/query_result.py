"""This module computes the results of sparql queries from DBpedia and saves the results in a json file.
Examples:
    $ python query_result.py --queryfile generated_queries.json --savefile result.json
    
"""

from SPARQLWrapper import SPARQLWrapper, JSON
import json
import sys
import argparse


def get_queries(input_file, out_file):

	endpoint = 'https://dbpedia.org/sparql'
	sp = SPARQLWrapper(endpoint)
	with open(input_file, 'r') as f:
		data = json.load(f)

	result = []
	for k,e in data.items():
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
	with open(out_file, "w") as f2:
		return json.dump(result, f2)

def main(args):
	get_queries(args.queryfile, args.savefile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queryfile',
                        help='Filepath where the sparql'
                        'query are stored.')
    parser.add_argument('--savefile',
                        help='Filepath where the query results'
                        'will be stored.')
    args = parser.parse_args()
    main(args)


