from amrqa.evaluate import evaluate_query


def test_perfect():
    ground_truth = set('ABC')
    predictions = set('ABC')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 1.0


def test_two_thirds():
    ground_truth = set('ABC')
    predictions = set('BCD')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 2 / 3


def test_no_predictions():
    ground_truth = set('ABC')
    predictions = set()
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 0


def test_no_ground_truth():
    ground_truth = set()
    predictions = set('ABC')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 0


def test_both_empty():
    ground_truth = set()
    predictions = set()
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert recall == precision == f1 == 1


def test_high_precision():
    ground_truth = set('ABC')
    predictions = set('A')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert precision == 1
    assert recall == 1 / 3
    assert f1 == (2 * 1 * (1 / 3)) / (1 + (1 / 3))


def test_high_recall():
    ground_truth = set('A')
    predictions = set('ABC')
    recall, precision, f1 = evaluate_query(ground_truth, predictions)
    assert precision == 1 / 3
    assert recall == 1
    assert f1 == (2 * (1 / 3) * 1) / ((1 / 3) + 1)
