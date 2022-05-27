from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from explainaboard.info import BucketCase, BucketCaseCollection
from explainaboard.utils.typing_utils import unwrap

T = TypeVar('T')

_INFINITE_INTERVAL = (-1e10, 1e10)


def find_key(dict_obj, x):
    for k, v in dict_obj.items():
        if len(k) == 1:
            if x == k[0]:
                return k
        elif len(k) == 2 and x >= k[0] and x <= k[1]:  # Attention !!!
            return k


def bucket_attribute_specified_bucket_value(
    sample_features: list[tuple[BucketCase, T]],
    bucket_number: int = 4,
    bucket_setting: Any = None,
) -> list[BucketCaseCollection]:
    if len(sample_features) == 0:
        return [BucketCaseCollection(_INFINITE_INTERVAL, [])]
    if bucket_setting is not None and len(bucket_setting) > 0:
        raise NotImplementedError(
            'bucket_setting incompatible with '
            'bucket_attribute_specified_bucket_value'
        )
    # Bucketing different Attributes
    cases = [x1 for x1, x2 in sample_features]
    vals = np.array([x2 for x1, x2 in sample_features])
    # Function to convert numpy datatypes to Python native types
    conv = int if np.issubdtype(vals[0], int) else float
    # Special case of one bucket
    if bucket_number == 1:
        max_val, min_val = conv(np.max(vals)), conv(np.min(vals))
        return [BucketCaseCollection((min_val, max_val), cases)]

    n_examps = len(vals)
    sorted_idxs = np.argsort(vals)
    sorted_vals = vals[sorted_idxs]
    max_val, min_val = conv(sorted_vals[-1]), conv(sorted_vals[0])

    start_val, last_val = min_val, min_val
    start_i, cutoff_i = 0, n_examps / float(bucket_number)
    bucket_collections: list[BucketCaseCollection] = []
    for i, val in enumerate(sorted_vals):
        # Return the final bucket
        if bucket_number - len(bucket_collections) == 1 or val == max_val:
            bucket_collections.append(
                BucketCaseCollection(
                    (conv(start_val), max_val),
                    [cases[j] for j in sorted_idxs[start_i:]],
                )
            )
            break
        # If the last value is not the same, maybe make a new bucket
        elif val != last_val:
            if i >= cutoff_i:
                bucket_collections.append(
                    BucketCaseCollection(
                        (conv(start_val), conv(last_val)),
                        [cases[j] for j in sorted_idxs[start_i:i]],
                    )
                )
                start_val = val
                start_i = i
                cutoff_i = i + (n_examps - i) / float(
                    bucket_number - len(bucket_collections)
                )
            last_val = val

    return bucket_collections


def bucket_attribute_discrete_value(
    sample_features: list[tuple[BucketCase, T]],
    bucket_number: int = int(1e10),
    bucket_setting: Any = 1,
) -> list[BucketCaseCollection]:
    """
    Bucket attributes by discrete value.
    :param sample_features: Pairs of a bucket case and feature value.
    :param bucket_number: Maximum number of buckets
    :param bucket_setting: Minimum number of examples per bucket
    """
    feat2case = {}
    for k, v in sample_features:
        if v not in feat2case:
            feat2case[v] = [k]
        else:
            feat2case[v].append(k)
    bucket_collections = [
        BucketCaseCollection((k,), v)
        for k, v in feat2case.items()
        if len(v) >= bucket_setting
    ]
    bucket_collections.sort(key=lambda x: -len(x.samples))
    if len(bucket_collections) > bucket_number:
        bucket_collections = bucket_collections[:bucket_number]
    return bucket_collections


def bucket_attribute_specified_bucket_interval(
    sample_features: list[tuple[BucketCase, T]],
    bucket_number: int,
    bucket_setting: list[tuple],
) -> list[BucketCaseCollection]:
    intervals = unwrap(bucket_setting)
    bucket2examp: dict[tuple, list[BucketCase]] = {k: list() for k in intervals}

    if isinstance(list(intervals)[0][0], str):  # discrete value, such as entity tags
        for k, v in sample_features:
            if v in bucket2examp:
                bucket2examp[(v,)].append(k)
    else:
        for examp, value in sample_features:
            res_key = find_key(bucket2examp, value)
            if res_key is None:
                continue
            bucket2examp[res_key].append(examp)

    bucket_collections = [
        BucketCaseCollection((k,), v) for k, v in bucket2examp.items()
    ]

    return bucket_collections
