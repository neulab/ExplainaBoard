from collections import Counter
import string
import re
from typing import List


'''
QA
'''


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_qa_sample_level(prediction: str, ground_truth: str):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_sample_level(prediction: str, ground_truth: str):
    # print("prediction, ground_truth 2: ",prediction, ground_truth)
    # print("normalize_answer: ",normalize_answer(prediction),normalize_answer(ground_truth))
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list):
    # scores_for_ground_truths = []
    # # for ground_truth in ground_truths:
    # score = metric_fn(prediction, ground_truths)
    # scores_for_ground_truths.append(score)
    # return max(scores_for_ground_truths)

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        # print(prediction, ground_truth)
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def exact_match_qa(true_answers: List[list], predicted_answer: List[str]):
    exact_match = 0
    total = 0
    # for k1,k2 in zip(true_Anss,pred_Anss):
    #     print(k1, k2)
    for true_ans, pred_ans in zip(true_answers, predicted_answer):
        total += 1
        exact_match1 = metric_max_over_ground_truths(
            exact_match_sample_level, pred_ans, true_ans
        )

        exact_match += exact_match1

    exact_match = 100.0 * exact_match / total

    return exact_match


def f1_score_qa(true_answers: List[list], predicted_answer: List[str]):
    f1_dataset_level = 0
    total = 0

    for true_ans, pred_ans in zip(true_answers, predicted_answer):
        total += 1

        f1_sample = metric_max_over_ground_truths(
            f1_score_qa_sample_level, pred_ans, true_ans
        )
        f1_dataset_level += f1_sample

    f1_dataset_level = 100.0 * f1_dataset_level / total

    return f1_dataset_level
