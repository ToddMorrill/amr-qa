"""This module retrieves SPARQL query results from the DBpedia API and saves
the results in a JSON file.

Examples:
    $ python query.py \
        --query-file generated_queries.json \
        --save-file result.json 
    
    $ python -m amrqa.query \
        --query-file ~/Documents/data/amr-qa/generate/v2/generated_queries.json \
        --save-file ~/Documents/data/amr-qa/generate/v2/generated_results.json
"""
import argparse
import json
import time

from SPARQLWrapper import JSON, SPARQLWrapper


def get_query_results(input_file, out_file):
    """Queries the DBPedia SPARQL endpoint with the queries in input_file and
    saves the results in out_file."""
    endpoint = 'https://dbpedia.org/sparql'
    sp = SPARQLWrapper(endpoint)
    with open(input_file, 'r') as f:
        data = json.load(f)

    result = []
    query_times = []
    for idx, item in enumerate(data.items()):
        start = time.time()
        k, e = item
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
        end = time.time()
        duration = end - start
        query_times.append(duration)
        if idx % 100 == 0:
            avg_query_time = sum(query_times) / len(query_times)
            print(
                f'Average query time at iteration {idx}: {avg_query_time:.2f}'
                ' seconds/query.')

    with open(out_file, "w") as f2:
        return json.dump(result, f2, indent=4)


def main(args):
    get_query_results(args.query_file, args.save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-file',
                        help='Filepath where the sparql'
                        'query are stored.')
    parser.add_argument('--save-file',
                        help='Filepath where the query results'
                        'will be stored.')
    args = parser.parse_args()
    main(args)
