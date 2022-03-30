import torch
import torchvision
import amrlib
import pandas as pd 
import sys
import json 

## location where the generated graphs should be stored
graph_output_dir = "data/qald9/parsed_graphs/"

## location of the data file
train_data_file = "data/qald9/qald-9-train-multilingual.json"

## load the sentence to graph (stog) model
stog = amrlib.load_stog_model()


## loads the english sentences from the data into a list
def load_sents():
    sents = []
    fp = open(train_data_file)
    dataset = json.load(fp)["questions"]
    for question in dataset:
        lang_list = question["question"]
        for lang in lang_list:
            if lang["language"]=="en":
                sents.append(lang["string"])
    return sents

## parses the sentences, writing resulting graph to a file
def parse_sents(sents):
    num_sents = len(sents)
    for idx, sentence in enumerate(sents):
        # if idx % 10 == 0:
        #     print_status(idx, num_sents)
        graph = stog.parse_sents([sentence])
        amr_output_file = graph_output_dir + str(idx+1) + ".txt"
        with open(amr_output_file, 'w') as f:
            for item in graph:
                f.write("%s\n" % item)

def print_status(current, total_count):
    percent_complete = str(round((current/total_count)*100, 2))
    print("Parsing sentence " + str(current) + " / " + str(total_count) + " ( " + percent_complete + " percent complete)")

def main():
    sents = load_sents()
    parse_sents(sents)

if __name__ == "__main__":
    main()