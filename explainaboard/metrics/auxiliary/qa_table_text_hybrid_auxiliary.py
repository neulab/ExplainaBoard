"""Pre-processing functions for qa-table_text_hybrid task.

The pre-processing is so complicated that I put most of the original processing code
here. The official evaluation script can be found here:
https://github.com/NExTplusplus/L2I/blob/main/evaluate.py
"""
# Because this is essentially third-party code, ignore linting errors
# flake8: noqa

from __future__ import annotations

import re
import string
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def negative_num_factor(x: str) -> float:
    """
    :param x:  transform (134) -> -134
    :return:
    """
    found = re.search(r"(\([\d.\s]+\))", x.strip())
    return -1.0 if found else 1.0


def percent_num_factor(x: str) -> float:
    """
    :param x:  transform 12% -> 12/100
    :return:
    """
    found = re.search(r"([\d.\s]+%)", x.strip())
    return 0.01 if found else 1.0


def word_scale_factor(x: str) -> float:
    """
    :param x: 1 million = 1_000_000
    :return:
    """
    found = re.search(r"([\d.]+\s?[a-zA-Z]+)", x)
    return scale_to_num(found.group(0).lower()) if found else 1.0


def scale_to_num(scale: str) -> float:
    scale = scale.lower()
    num = 1.0
    if "hundred" in scale:  # hundred
        num = 100.0
    elif "thousand" in scale:  # thousand
        num = 1000.0
    elif "million" in scale:  # million
        num = 1000000.0
    elif "billion" in scale:  # billion
        num = 1000000000.0
    elif "percent" in scale:  # percent
        num = 0.01
    return num


def extract_one_num_from_str(s: str) -> int | None | float:
    s = _clean_num(s)
    r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
    groups = re.findall(r_num, s)
    if len(groups) == 0:
        return None
    num = groups[0][0]
    if num == "":
        return None
    if "." in num:
        return float(num)
    return int(num)


EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"


def _clean_num(text: str) -> str:
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def is_number(text: str) -> bool:
    try:
        words = " ".join([_clean_num(w) for w in text.split()]).split()
        if len(words) == 0:
            """1023 or 1 million"""
            return False
        num = float(words[0])
        if np.isnan(num):
            return False
        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False
        return True
    except ValueError:
        return False


def to_number(text: str) -> float | None:
    num = extract_one_num_from_str(text)
    scale_val = word_scale_factor(text)
    negative_flag = negative_num_factor(text)
    percent_flag = percent_num_factor(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def remove_punc(text: str) -> str:
    if not is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def lower(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list[str]:
    return re.split(" ", text)


def normalize_number(text: str) -> str:
    if is_number(text):
        return str(to_number(text))
    else:
        return text


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    parts = [
        white_space_fix(remove_articles(normalize_number(remove_punc(lower(token)))))
        for token in tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])


def ws_tokenize(text: str) -> list[str]:
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip().lower()
    if not text:
        return []
    text = white_space_fix(text)
    tokens = text.split()
    tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
    return tokens


def _answer_to_bags(
    answer: str | list[str] | tuple[str, ...]
) -> tuple[list[str], list[set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: list[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: list[set[str]], gold: list[set[str]]) -> list[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # if _match_numbers_if_present(gold_item, pred_item): no
            # need to match number in tatqa
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: set[str], gold_bag: set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: set[str], predicted_bag: set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def extract_gold_answers(qa_annotation: dict[str, Any]) -> tuple[str, list, str]:
    """
    span
    multi-span
    arithmetic (+ - * /)
    count
    date
    other
    gold answers is a list of list, each item in gold answers is a valid answer
    """
    answer_type, scale = qa_annotation["answer_type"], qa_annotation["scale"]
    answer_content = qa_annotation["answer"]
    gold_answers = []
    if answer_type in ["multi-span", "span"]:  # list
        if not isinstance(answer_content, list):
            raise TypeError("answer_content must be list.")
        gold_answers = answer_content  # multi-span
    elif answer_type in ["arithmetic"]:
        gold_answers.append(str(answer_content))
    elif answer_type in ["count"]:
        gold_answers.append(str(int(answer_content)))
    else:
        gold_answers.append(str(answer_content))
    return answer_type, gold_answers, scale


def get_answer_str(answers: list[str], scale: str) -> list[str]:
    """
    :param ans_type:  span, multi-span, arithmetic, count
    :param ans_list:
    :param scale: "", thousand, million, billion, percent
    :param mode:
    :return:

    """
    sorted_ans = sorted(answers)
    ans_temp = []
    for ans in sorted_ans:
        ans_str = str(ans)
        if is_number(ans_str):
            ans_num = to_number(ans_str)
            if ans_num is None:
                if scale:
                    ans_str = ans_str + " " + str(scale)
            else:
                if "%" in ans_str:
                    ans_str = "%.4f" % ans_num
                else:
                    ans_str = "%.4f" % (round(ans_num, 2) * scale_to_num(scale))
        else:
            if scale:
                ans_str = ans_str + " " + str(scale)
        ans_temp.append(ans_str)
    return [" ".join(ans_temp)]


def add_percent_pred(
    prediction_strings: list[str], pred_scale: str, pred: list[str]
) -> list[str]:
    """
    to solve [pred = 0.2342] <>   [ans = 23.42 and scale == 'percent']

    :param prediction_strings:
    :param gold_ans_type:
    :param gold_scale:
    :param pred:
    :return:
    """
    if len(pred) != 1:
        return prediction_strings
    pred_str = str(pred[0])
    if pred_str is None:
        return prediction_strings
    if (
        not pred_scale and "%" not in pred_str and is_number(pred_str)
    ):  # mode only or no pred_scale num only
        pred_str_num = to_number(pred_str)
        if pred_str_num is None:
            return prediction_strings
        prediction_strings.append("%.4f" % pred_str_num)
    return prediction_strings
