"""
    This file performs parsing and linking of an entire dataset.
    


"""

import blink.main_dense as main_dense
import amrlib
import sys
import json
import os
from os.path import exists
import argparse
from wiki_adder_blink import WikiAdderBlink
from IPython.display import clear_output

GTOS_MODEL = amrlib.load_gtos_model()
STOG_MODEL = amrlib.load_stog_model()

def load_blink_models(blink_model_dir, score_thresh):
    """ This function loads blink models from a given directory and 
        creates the WikiAdderBlink object with given score threshold. """
    wiki_adder = WikiAdderBlink(blink_model_dir, score_thresh=score_thresh)
    return wiki_adder

def load_dataset(data_path):
    """ This function loads the dataset to be linked, returning it's JSON content"""
    fp = open("../data/qald_9_train.json")
    qald9_content = json.load(fp)
    fp.close()
    return qald9_content

def link_data(json_content, wiki_adder):
    json_length = len(json_content)
    counter = 0
    new_json = dict()
    for elem in json_content:
        new_json[elem] = dict()
        data_pt = json_content[elem]
        q_text = data_pt["text"]
        
        # ensure there is separation between text and punctuation, required by blink
        q_text = q_text[:-1] + " " + q_text[-1]
        new_json[elem]["text"] = q_text

        # parse sentence into an AMR graph
        graph = STOG_MODEL.parse_sents([q_text])[0]

        # check whether graph has any named nodes - if not, no entities will be linked
        if graph.find(":name") == -1:
            #ignore sentence, only want graph
            idx = graph.find('(')
            new_json[elem]["amr"] = graph[idx:]
        else:
            # wiki adder takes file input, so we create some temp files to write/read
            temp_graph_file = "temp_graph.txt"
            temp_linked_file = "temp_linked.txt"
            temp_graph_file.write("%s\n" % graph)
            temp_graph_file.close()
            wiki_adder.wikify_file(temp_graph_file, temp_linked_file)
            linked_graph_content = open(temp_linked_file, "r")
            next(linked_graph_content)
            linked_graph = linked_graph_content.read()
            new_json[elem]["amr"] = linked_graph
            linked_graph_content.close()

        # simple progress bar
        counter+=1
        clear_output(wait=True)
        print(str(counter) + " / " + str(json_length))
    return new_json

def main(args):
    blink_model_path = args.blink_models_path
    score_thresh = args.score_thresh
    wiki_adder = load_blink_models(blink_model_path, score_thresh)
    data_path = args.data_filepath
    data_json = load_dataset(data_path)
    updated_json = link_data(data_json, wiki_adder)
    output_path = args.output_filepath
    with open(output_path, "w") as fp:
        json.dump(updated_json,fp) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blink-models-path',
                        help='Filepath where the BLINK models are stored.')
    parser.add_argument(
        '--data-filepath',
        help='Filepath where the dataset is stored.')
    parser.add_argument(
        '--score-thresh',
        help='Score threshold to be used for BLINK candidate selection.')
    parser.add_argument(
        '--output-filepath',
        help='Filepath to where the outputted data should bes')
    args = parser.parse_args()
    main(args)