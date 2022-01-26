import numpy as np
import os
import scipy.stats as statss
import scipy
from random import choices
from seqeval.metrics import precision_score, recall_score, f1_score
from collections import Counter
import string
import re
import argparse
import json
import sys
from typing import Any, ClassVar, Dict, List, Optional


'''
Sequence Labeling
'''
def f1_score_seqeval(labels, predictions, language=None):
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return {'f1': f1 * 100, 'precision': precision * 100, 'recall': recall * 100}


def get_chunks(seq):
    """
    tags:dic{'per':1,....}
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = 'O'
    # idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_chunk_type(tok):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tok_split = tok.split('-')
    return tok_split[0], tok_split[-1]



def accuracy(labels:List[str], predictions:List[str], language=None):
    correct = sum([int(p == l) for p, l in zip(predictions, labels)])
    accuracy_value = float(correct) / len(predictions)
    return accuracy_value * 100


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h



def compute_confidence_interval_acc(true_label_list, pred_label_list, n_times=1000):

    def get_sample_rate(n_data):
        res = 0.8
        if n_data > 300000:
            res = 0.1
        elif n_data > 100000 and n_data < 300000:
            res = 0.2

        return res

    n_data = len(true_label_list)
    sample_rate = get_sample_rate(n_data)
    n_sampling = int(n_data * sample_rate)
    if n_sampling == 0:
        n_sampling = 1
    # print("n_data:\t", n_data)
    # print("sample_rate:\t", sample_rate)
    # print("n_sampling:\t", n_sampling)

    performance_list = []
    confidence_low, confidence_up = 0, 0
    for i in range(n_times):
        sample_index_list = choices(range(n_data), k=n_sampling)

        performance = accuracy(list(np.array(true_label_list)[sample_index_list]),
                               list(np.array(pred_label_list)[sample_index_list]))
        performance_list.append(performance)

    if n_times != 1000:
        confidence_low, confidence_up = mean_confidence_interval(performance_list)
    else:
        performance_list.sort()
        confidence_low = performance_list[24]
        confidence_up = performance_list[974]

    # print("\n")
    # print("confidence_low:\t", confidence_low)
    # print("confidence_up:\t", confidence_up)

    return confidence_low, confidence_up