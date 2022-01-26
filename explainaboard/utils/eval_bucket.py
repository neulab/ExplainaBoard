from .eval_basic import *
from .py_utils import *


def f1_score_seqeval_bucket(pred_chunks, true_chunks):

    correct_preds, total_correct, total_preds = 0., 0., 0.
    correct_preds = len(set(true_chunks) & set(pred_chunks))
    total_preds = len(pred_chunks)
    total_correct = len(true_chunks)


    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    # acc = np.mean(accs)
    return f1, p, r




