import torch
import torchvision
import amrlib
import pandas as pd 
import sys
import json 
import argparse

## location of the data file
train_data_file_qald = "data/qald-9-train-multilingual.json"
train_data_file_lcquad = "data/lcquad-train.json"
## load the sentence to graph (stog) model
stog = amrlib.load_stog_model()


## loads the sentences lcquad from the data into a list of tuples (id, sentence)
def load_sents_lc_quad():
    fp = open(train_data_file_lcquad)
    data = json.load(fp)
    list_id_sent = [(sent["_id"], sent["corrected_question"]) for sent in data]
    print(list_id_sent[0])
    return list_id_sent

## loads the english sentences from qald into a list of tuples (id, sentence)
def load_sents_qald():
    sents = []
    fp = open(train_data_file_qald)
    dataset = json.load(fp)["questions"]
    for question in dataset:
        id = question["id"]
        lang_list = question["question"]
        for lang in lang_list:
            if lang["language"]=="en":
                sents.append((id, lang["string"]))
    return sents

## parses the sentences, writing resulting graph to a file
def parse_sents(sents, graph_output_dir):
    num_sents = len(sents)
    for idx, sentence in enumerate(sents):
        sentence_id = sentence[0]
        sentence_lit = sentence[1]
        if idx % 10 == 0:
            print_status(idx, num_sents)
        graph = stog.parse_sents([sentence_lit])
        amr_output_file = graph_output_dir + sentence_id + ".txt"
        with open(amr_output_file, 'w') as f:
            for item in graph:
                f.write("%s\n" % item)

def print_status(current, total_count):
    percent_complete = str(round((current/total_count)*100, 2))
    print("Parsing sentence " + str(current) + " / " + str(total_count) + " ( " + percent_complete + " percent complete)")

def main(args):
    sentences = []
    if args.dataset == 'lcquad':
        sentences= load_sents_lc_quad()
        parse_sents(sentences, "data/lcquad/parsed_graphs/")
    elif args.dataset == 'qald9':
        sentences = load_sents_qald()
        parse_sents(sentences, "data/qald9/parsed_graphs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', 
        help='The dataset to parse. Either `lcquad` or `qald9')
    args = parser.parse_args()
    main(args)