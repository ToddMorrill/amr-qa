"""This module computes evaluation metrics by comparing the query results
retrieved for the ground-truth queries against the query results retrieved for
the generated queries.

Examples:
    $ python evaluate.py
"""
import argparse
import json

from pyparsing import opAssoc

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_query(ground_truth, predictions):
    # Measure the following for a single query:
    # Recall = (# correct answers) / (# ground-truth answers)
    # Precision = (# correct answers) / (# predicted answers)
    # F1 = (2 * Precision * Recall) / (Precision + Recall)

    # if both ground-truth and predictions are empty, then full marks
    if (len(ground_truth) == 0) and (len(predictions) == 0):
        recall = 1
        precision = 1
        f1 = 1
    # otherwise determine how many of the predictions are correct
    else:
        correct = 0
        for pred in predictions:
            if pred in ground_truth:
                correct += 1

        # avoid divide by 0
        if len(ground_truth) > 0:
            recall = correct / len(ground_truth)
        else:
            recall = 0
        if len(predictions) > 0:
            precision = correct / len(predictions)
        else:
            precision = 0
        
        denominator = recall + precision
        if denominator > 0:
            f1 = 2*recall*precision/denominator
        else:
            f1 = 0
    return recall, precision, f1

def evaluate(ground_truth, predictions):
    if len(ground_truth) != len(predictions):
        raise ValueError('Length of ground truth must equal the length of the'
        'predictions.')
    # Macro F1
    # Measure precision, recall, and F1 for each query and average the results
    recall = 0
    precision = 0
    f1 = 0
    for i in range(len(ground_truth)):
        query_ground_truth, query_predictions = ground_truth[i], predictions[i]
        result = evaluate_query(query_ground_truth, query_predictions)
        query_recall, query_precision, query_f1 = result
        recall += query_recall
        precision += query_precision
        f1 += query_f1

    # Calcuate averages
    recall = recall / len(ground_truth)
    precision = precision / len(ground_truth)
    f1 = f1 / len(ground_truth)
    return recall, precision, f1

def test_perfect():
    ground_truth = set('ABC')
    predictions = set('ABC')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 1.0
    
def test_two_thirds():
    ground_truth = set('ABC')
    predictions = set('BCD')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 2/3

def test_no_predictions():
    ground_truth = set('ABC')
    predictions = set()
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 0

def test_no_ground_truth():
    ground_truth = set()
    predictions = set('BCD')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 0

def both_empty():
    ground_truth = set()
    predictions = set()
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 1

def high_precision():
    ground_truth = set('ABC')
    predictions = set('A')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert precision == 1
    assert recall == 1/3
    assert f1 == (2*1*(1/3)) / (1 + (1/3))

def high_recall():
    ground_truth = set('A')
    predictions = set('ABC')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert precision == 1/3
    assert recall == 1
    assert f1 == (2*(1/3)*1) / ((1/3) + 1)

def main(args):
    # ground_truth = load_json(args.ground_truth)
    # predictions = load_json(args.predictions)
    
    # preprocess ground-truth results from ground-truth queries against DBPedia 2016

    # preprocess ground-truth results from ground-truth queries against current (2022) DBPedia

    # preprocess ground-truth values from predicted queries against current (2022) DBPedia
    test_perfect()
    test_two_thirds()
    test_no_predictions()
    test_no_ground_truth()
    both_empty()
    high_precision()
    high_recall()
    breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', help='Filepath where the ground-truth'
    'query results are stored.')
    parser.add_argument('--predictions', help='Filepath where the query results'
    'for the predicted queries are stored.')
    args = parser.parse_args()
    main(args)