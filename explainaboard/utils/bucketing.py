from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from explainaboard.utils.analysis import find_key, reverse_dict
from explainaboard.utils.py_utils import sort_dict

T = TypeVar('T')


def bucket_attribute_specified_bucket_value(
    dict_obj: dict[Any, T], bucket_number: int = 4, bucket_setting: Any = None
) -> dict[tuple[T, T], list]:
    if len(dict_obj) == 0:
        return {}
    if bucket_setting is not None and len(bucket_setting) > 0:
        raise NotImplementedError(
            'bucket_setting incompatible with '
            'bucket_attribute_specified_bucket_value'
        )
    # Bucketing different Attributes
    keys = list(dict_obj.keys())
    vals = np.array(list(dict_obj.values()))
    # Function to convert numpy datatypes to Python native types
    conv = int if np.issubdtype(vals[0], int) else float
    # Special case of one bucket
    if bucket_number == 1:
        max_val, min_val = conv(np.max(vals)), conv(np.min(vals))
        return {(min_val, max_val): keys}

    n_examps = len(keys)
    sorted_idxs = np.argsort(vals)
    sorted_vals = vals[sorted_idxs]
    max_val, min_val = conv(sorted_vals[-1]), conv(sorted_vals[0])

    start_val, last_val = min_val, min_val
    start_i, cutoff_i = 0, n_examps / float(bucket_number)
    bucket_dict: dict[tuple[T, T], list[Any]] = {}
    for i, val in enumerate(sorted_vals):
        # Return the final bucket
        if bucket_number - len(bucket_dict) == 1 or val == max_val:
            bucket_dict[(conv(start_val), max_val)] = [
                keys[j] for j in sorted_idxs[start_i:]
            ]
            break
        # If the last value is not the same, maybe make a new bucket
        elif val != last_val:
            if i >= cutoff_i:
                bucket_dict[(conv(start_val), conv(last_val))] = [
                    keys[j] for j in sorted_idxs[start_i:i]
                ]
                start_val = val
                start_i = i
                cutoff_i = i + (n_examps - i) / float(bucket_number - len(bucket_dict))
            last_val = val

    return bucket_dict


def bucket_attribute_discrete_value(
    dict_obj=None,
    bucket_number=100000000,
    bucket_setting=1,
):
    # Bucketing different Attributes
    dict_span2att_val = dict_obj
    n_buckets = bucket_number
    n_entities = bucket_setting

    dict_bucket2span = {}

    dict_att_val2span = reverse_dict(dict_span2att_val)
    dict_att_val2span = sort_dict(dict_att_val2span, flag="value")

    n_total = 1
    for att_val, entity in dict_att_val2span.items():

        if len(entity) < n_entities or n_total > n_buckets:
            break
        dict_bucket2span[(att_val,)] = entity

        n_total += 1

    return dict_bucket2span


def bucket_attribute_specified_bucket_interval(
    dict_obj=None,
    bucket_number=None,
    bucket_setting=None,
):
    # Bucketing different Attributes

    # hardcoded_bucket_values = [set([float(0), float(1)])]

    # intervals = [0, (0,0.5], (0.5,0.9], (0.99,1]]
    dict_span2att_val = dict_obj
    intervals = bucket_setting

    dict_bucket2span = {}
    if isinstance(list(intervals)[0][0], str):  # discrete value, such as entity tags
        dict_att_val2span = reverse_dict(dict_span2att_val)
        dict_att_val2span = sort_dict(dict_att_val2span, flag="value")
        for att_val, entity in dict_att_val2span.items():
            att_val_tuple = (att_val,)
            if att_val_tuple in intervals:
                if att_val_tuple not in dict_bucket2span.keys():
                    dict_bucket2span[att_val_tuple] = entity
                else:
                    dict_bucket2span[att_val_tuple] += entity

        for val in intervals:
            if val not in dict_bucket2span.keys():
                dict_bucket2span[val] = []
    else:
        dict_att_val2span = reverse_dict(dict_span2att_val)
        dict_att_val2span = sort_dict(dict_att_val2span)
        for v in intervals:
            if len(v) == 1:
                dict_bucket2span[v] = []
            else:
                dict_bucket2span[v] = []

        for att_val, entity in dict_att_val2span.items():
            res_key = find_key(dict_bucket2span, att_val)
            if res_key is None:
                continue
            dict_bucket2span[res_key] += entity

    return dict_bucket2span
