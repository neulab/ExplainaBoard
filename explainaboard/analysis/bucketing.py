"""Implements methods for bucketing."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

# List and Tuple is required for the first argument of narrow().
from typing import Any, cast, List, Tuple

import numpy as np

from explainaboard.analysis.case import AnalysisCase, AnalysisCaseCollection

_INFINITE_INTERVAL = (-1e10, 1e10)


def find_range(
    keys: Iterable[tuple[float, float]], x: float
) -> tuple[float, float] | None:
    """Finds the range that covers the given value.

    Args:
        keys: iterable of ranges.
        x: target value.

    Returns:
        The range `k` in `keys` that satisfies `k[0] <= x <= k[1]`, or None if there are
        no such range in `keys`.
    """
    return next(filter(lambda k: k is not None and k[0] <= x <= k[1], keys), None)


def continuous(
    sample_features: list[tuple[AnalysisCase, Any]],
    bucket_number: int = 4,
    bucket_setting: Any = None,
) -> list[AnalysisCaseCollection]:
    """Bucketing based on continuous features.

    Takes in examples and attempts to split them into `bucket_number` approximately
    equal-sized buckets.

    Args:
        sample_features: A list of tuples including an analysis case, and a feature
          value.
        bucket_number: The number of buckets to generate.
        bucket_setting: Not used by this bucketing method, so it will fail if this is
          set to anything other than none.

    Returns:
        A list of AnalysisCaseCollections corresponding to the buckets.
    """
    if len(sample_features) == 0:
        return [AnalysisCaseCollection(samples=[], interval=_INFINITE_INTERVAL)]
    if bucket_setting is not None and len(bucket_setting) > 0:
        raise NotImplementedError('bucket_setting incompatible with continuous')
    # Bucketing different Attributes
    cases = [x1 for x1, x2 in sample_features]
    vals = np.array([x2 for x1, x2 in sample_features])
    # Function to convert numpy datatypes to Python native types
    conv = int if np.issubdtype(type(vals[0]), int) else float
    # Special case of one bucket
    if bucket_number == 1:
        max_val, min_val = conv(np.max(vals)), conv(np.min(vals))
        return [
            AnalysisCaseCollection(
                samples=list(range(len(cases))),
                interval=(min_val, max_val),
            )
        ]

    n_examps = len(vals)
    sorted_idxs = np.argsort(vals)
    sorted_vals = vals[sorted_idxs]
    max_val, min_val = conv(sorted_vals[-1]), conv(sorted_vals[0])

    start_val, last_val = min_val, min_val
    start_i, cutoff_i = 0, n_examps / float(bucket_number)
    bucket_collections: list[AnalysisCaseCollection] = []
    for i, val in enumerate(sorted_vals):
        # If the last value is not the same, maybe make a new bucket
        if val != last_val:
            if i >= cutoff_i:
                bucket_collections.append(
                    AnalysisCaseCollection(
                        samples=[int(j) for j in sorted_idxs[start_i:i]],
                        interval=(conv(start_val), conv(last_val)),
                    )
                )
                start_val = val
                start_i = i
                cutoff_i = i + (n_examps - i) / float(
                    bucket_number - len(bucket_collections)
                )
            last_val = val
    # Return the last bucket
    bucket_collections.append(
        AnalysisCaseCollection(
            samples=[int(j) for j in sorted_idxs[start_i:]],
            interval=(conv(start_val), max_val),
        )
    )

    return bucket_collections


def discrete(
    sample_features: list[tuple[AnalysisCase, Any]],
    bucket_number: int = int(1e10),
    bucket_setting: Any = 1,
) -> list[AnalysisCaseCollection]:
    """Bucket attributes by discrete value.

    It will return buckets for the `bucket_number` most frequent discrete values.

    Args:
        sample_features: Pairs of a analysis case and feature value.
        bucket_number: Maximum number of buckets
        bucket_setting: Minimum number of examples per bucket

    Returns:
        A list of AnalysisCaseCollections corresponding to the buckets.
    """
    feat2idx = {}
    if bucket_setting is None:
        bucket_setting = 0
    for idx, (case, feat) in enumerate(sample_features):
        if feat not in feat2idx:
            feat2idx[feat] = [idx]
        else:
            feat2idx[feat].append(idx)
    bucket_collections = [
        AnalysisCaseCollection(samples=idxs, name=feat)
        for feat, idxs in feat2idx.items()
        if len(idxs) >= bucket_setting
    ]
    bucket_collections.sort(key=lambda x: -len(x.samples))
    if len(bucket_collections) > bucket_number:
        bucket_collections = bucket_collections[:bucket_number]
    return bucket_collections


def fixed(
    sample_features: list[tuple[AnalysisCase, Any]],
    bucket_number: int,
    bucket_setting: Any,
) -> list[AnalysisCaseCollection]:
    """Bucketing based on pre-determined buckets.

    Args:
        sample_features: A list of tuples including an analysis case, and a feature
          value.
        bucket_number: Ignored by this function.
        bucket_setting: A list of bucket names or intervals, depending on the type.

    Returns:
        A list of AnalysisCaseCollections corresponding to the buckets.
    """
    interval_or_names = cast(List[Hashable], bucket_setting)
    if len(interval_or_names) == 0:
        raise ValueError("Can not determine bucket keys.")

    features = [x[1] for x in sample_features]

    if isinstance(interval_or_names[0], str):
        names = cast(List[str], interval_or_names)
        name2idx: dict[str, list[int]] = {k: [] for k in names}
        name_features = cast(List[str], features)

        for idx, name in enumerate(name_features):
            if name in names:
                name2idx[name].append(idx)

        return [AnalysisCaseCollection(samples=v, name=k) for k, v in name2idx.items()]
    else:
        intervals = cast(List[Tuple[float, float]], interval_or_names)
        interval2idx: dict[tuple[float, float], list[int]] = {k: [] for k in intervals}
        interval_features = cast(List[float], features)

        for idx, interval in enumerate(interval_features):
            key = find_range(intervals, interval)
            if key is not None:
                interval2idx[key].append(idx)

        return [
            AnalysisCaseCollection(samples=v, interval=k)
            for k, v in interval2idx.items()
        ]
