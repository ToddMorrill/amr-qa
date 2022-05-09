"""This module computes evaluation metrics by comparing the query results
retrieved for the ground-truth queries against the query results retrieved for
the generated queries.

Examples:
    $ python -m amrqa.evaluate \
        --ground-truth ~/Documents/data/amr-qa/evaluate/qald-9-train-multilingual.json \
        --predictions ~/Documents/data/amr-qa/evaluate/qald9_result.json \
        --save-dir ~/Documents/data/amr-qa/evaluate/groundtruth16-vs-groundtruth22
    
    $ python -m amrqa.evaluate \
        --ground-truth ~/Documents/data/amr-qa/evaluate/qald9_result.json \
        --predictions ~/Documents/data/amr-qa/generate/v2/generated_results.json \
        --save-dir ~/Documents/data/amr-qa/evaluate/v2
"""
import argparse
import os
import json


def load_json(filepath):
    """Loads JSON file from the specified filepath."""
    with open(filepath, 'r') as f:
        return json.load(f)

class SetEncoder(json.JSONEncoder):
    """Custom JSON encoder to serialize sets to lists."""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def write_json(obj, filepath):
    """Saves the object to the specified filepath."""
    with open(filepath, 'w') as f:
        json.dump(obj, f, cls=SetEncoder, indent=4)


def evaluate_query(ground_truth, predictions):
    """Measures the following for a single query:
        - Recall = (# correct answers) / (# ground-truth answers)
        - Precision = (# correct answers) / (# predicted answers)
        - F1 = (2 * Precision * Recall) / (Precision + Recall)

    This follows the specifications in: http://ceur-ws.org/Vol-2241/paper-06.pdf
    """
    recall = 0
    precision = 0
    f1 = 0
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
        if len(predictions) > 0:
            precision = correct / len(predictions)

        denominator = recall + precision
        if denominator > 0:
            f1 = (2 * recall * precision) / denominator
    return recall, precision, f1


def evaluate(ground_truth, predictions, return_individual_scores=True):
    """Computes recall, precision, and F1 scores for each query and averages
    the results to compute macro scores. Optionally returns detailed
    instance-level scores."""
    if len(ground_truth) != len(predictions):
        raise ValueError('Length of ground truth must equal the length of the'
                         'predictions.')
    # Macro-scores
    # Measure recall, precision, and F1 for each query and average the results
    recalls = []
    precisions = []
    f1s = []
    for i in range(len(ground_truth)):
        query_ground_truth, query_predictions = ground_truth[i], predictions[i]
        result = evaluate_query(query_ground_truth, query_predictions)
        query_recall, query_precision, query_f1 = result
        recalls.append(query_recall)
        precisions.append(query_precision)
        f1s.append(query_f1)

    # Calcuate averages
    recall = sum(recalls) / len(ground_truth)
    precision = sum(precisions) / len(ground_truth)
    f1 = sum(f1s) / len(ground_truth)
    scores = {'recall': recall, 'precision': precision, 'f1': f1}
    if return_individual_scores:
        individual_scores = {
            'individual_recall': recalls,
            'individual_precision': precisions,
            'individual_f1': f1s
        }
        scores.update(individual_scores)
    return scores


def get_variable_bindings(answer):
    """Utility function to extract the actual answers to the query from the
    SPARQL results."""
    bindings_set = set()
    if 'results' not in answer:
        return set()
    if 'bindings' not in answer['results']:
        return set()
    for resource in answer['results']['bindings']:
        # TODO: is it possible to have the same binding for different variables
        # retrieve the bindings, ignore the variable names
        for var in resource.keys():
            value = resource[var]['value']
            bindings_set.add(value)
    return bindings_set


def preprocess_dbpedia16(ground_truth):
    """Preprocessing function for results that have answers from the 2016
    version of DBPedia (e.g. QALD 9)."""
    ground_truth_results = []
    for question in ground_truth['questions']:
        ground_truth_set = set()
        for answer in question['answers']:
            bindings_set = get_variable_bindings(answer)
            ground_truth_set.update(bindings_set)
        ground_truth_results.append(ground_truth_set)
    return ground_truth_results


def preprocess_dbpedia22(predictions):
    """Preprocessing function for results that have answers from the 2022
    version of DBPedia (e.g. QALD 9)."""
    predictions_results = []
    for answer in predictions:
        bindings_set = get_variable_bindings(answer)
        predictions_results.append(bindings_set)
    return predictions_results

def save_evaluation(ground_truth, predictions, metrics, save_dir):
    """Saves the side-by-side ground-truth and predictions in a JSON file and
    also saves the metrics dictionary in a JSON file."""
    save_dict = {}
    for i in range(len(ground_truth)):
        save_dict[i] = {'ground_truth': ground_truth[i], 'prediction': predictions[i]}
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'ground_truth_vs_predictions.json')
    write_json(save_dict, filepath)

    metrics_filepath = os.path.join(save_dir, 'metrics.json')
    write_json(metrics, metrics_filepath)


def main(args):
    ground_truth = load_json(args.ground_truth)
    predictions = load_json(args.predictions)

    if isinstance(ground_truth, dict):
        # preprocess ground-truth results from ground-truth queries against DBPedia 2016
        ground_truth_results = preprocess_dbpedia16(ground_truth)
    else:
        # preprocess ground-truth results from ground-truth queries against DBPedia 2022
        ground_truth_results = preprocess_dbpedia22(ground_truth)

    # preprocess results from queries against current (2022) DBPedia
    predictions_results = preprocess_dbpedia22(predictions)

    # compute metrics
    metrics = evaluate(ground_truth_results, predictions_results)
    print(
        f"Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}, F1: {metrics['f1']:.4f}"
    )

    # save results
    save_evaluation(ground_truth_results, predictions_results, metrics, args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth',
                        help='Filepath where the ground-truth'
                        'query results are stored.')
    parser.add_argument('--predictions',
                        help='Filepath where the query results'
                        'for the predicted queries are stored.')
    parser.add_argument('--save-dir',
                        help='Directory where the results will be saved.')
    args = parser.parse_args()
    main(args)