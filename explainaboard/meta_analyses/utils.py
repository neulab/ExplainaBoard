"""Utility functions for performing meta-analyses."""

from __future__ import annotations

from typing import Any

from explainaboard.analysis.analyses import BucketAnalysisResult
from explainaboard.info import SysOutputInfo
from explainaboard.utils.typing_utils import narrow, unwrap


def report_to_sysout(report: SysOutputInfo) -> list[dict]:
    """Loops through all the buckets in a report, converts them to "examples".

    This is to mimic a system output file.

    The metrics that describe each bucket become the "features" of this new
    system output.
    """
    results_fine_grained = [
        narrow(BucketAnalysisResult, x)
        for x in unwrap(report.results.analyses)
        if isinstance(x, BucketAnalysisResult)
    ]
    meta_examples = []
    for feature_buckets in results_fine_grained:

        # feature_perfs has `n_buckets` elements, each corresponding to a single bucket
        for bucket in feature_buckets.bucket_performances:

            # loop through and record all the metrics that describe this bucket
            example_features: dict[str, Any] = {}
            for perf in bucket.performances:

                example_features['feature_name'] = feature_buckets.name
                example_features['bucket_interval'] = bucket.bucket_interval
                example_features['bucket_name'] = bucket.bucket_name
                example_features['bucket_size'] = bucket.n_samples
                example_features[perf.metric_name] = perf.value
                # example_features[f'{perf.metric_name}_CI'] = \
                # [perf.confidence_score_low, perf.confidence_score_high]

            meta_examples.append(example_features)
    return meta_examples
