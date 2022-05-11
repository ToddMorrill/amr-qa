"""
    This file does a spot check of N examples of the data,
    printing entity linking results of ground truth data vs
    parsed data.

    Usage: python spotcheck.py --gt-filepath ../data/qald9-train.txt 
                               --p-filepath ../data/qald9-parsed.txt
"""

import json
import random
import argparse

random.seed(5)
N_EXAMPLES = 30

def get_random_data(n, json_path):
    """
    This methods selects n examples from either ground truth data or parsed data.
    """
    fp = open(json_path)
    lcquad_content = json.load(fp)
    lcquad_train = {}
    for i in lcquad_content:
        if i.startswith("train"):
            lcquad_train[i] = lcquad_content[i]

    r = [random.choice(list(lcquad_train)) for i in range(50)]
    selected_vals = {key: lcquad_content[key] for key in r}
    return selected_vals

def main(args):
    gt_filepath = args.gt_filepath
    p_filepath = args.p_filepath
    gt_rand = get_random_data(N_EXAMPLES, gt_filepath)
    p_rand = get_random_data(N_EXAMPLES, p_filepath)
    for elem in gt_rand:
        print(elem)
        print("GT PARSE:\n")
        print(gt_rand[elem]["extended_amr"] + "\n")
        print("P PARSE:\n")
        print(p_rand[elem]["extended_amr"] + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-filepath',
                        help='Filepath where the ground truth JSON data is stored.')
    parser.add_argument(
        '--p-filepath',
        help='Filepath where JSON data (with our parsed AMR graphs) is stored.')
    args = parser.parse_args()
    main(args)