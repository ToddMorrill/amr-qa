def precision_recall_f1(predictions, golds):
    p, r, f1 = 0.0, 0.0, 0.0
    if len(predictions) > 0 and len(golds) > 0:
        p = (len(set(predictions) & set(golds))) / len(set(predictions))
        r = (len(set(predictions) & set(golds))) / len(golds)
        if p + r > 0:
            f1 = f1_score(p, r)
    return p, r, f1

def f1_score(p, r):
    if p + r == 0:
        return 0
    return 2 * ((p * r) / (p + r))