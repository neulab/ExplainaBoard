"""Utility functions for performing meta-analyses."""

from __future__ import annotations

from typing import Any, cast

from explainaboard.analysis.analyses import BucketAnalysisDetails
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.metric import Score


def report_to_sysout(report: SysOutputInfo) -> list[dict]:
    """Loops through all the buckets in a report, converts them to "examples".

    This is to mimic a system output file.

    The metrics that describe each bucket become the "features" of this new
    system output.
    """
    results_fine_grained = [
        x
        for x in report.results.analyses
        if isinstance(x.details, BucketAnalysisDetails)
    ]
    meta_examples = []
    for result in results_fine_grained:
        details = cast(BucketAnalysisDetails, result.details)

        # feature_perfs has `n_buckets` elements, each corresponding to a single bucket
        for bucket in details.bucket_performances:

            # loop through and record all the metrics that describe this bucket
            example_features: dict[str, Any] = {}
            for metric_name, metric_result in bucket.results.items():

                example_features["feature_name"] = result.name
                example_features["bucket_interval"] = bucket.bucket_interval
                example_features["bucket_name"] = bucket.bucket_name
                example_features["bucket_size"] = bucket.n_samples
                example_features[metric_name] = metric_result.get_value(
                    Score, "value"
                ).value

            meta_examples.append(example_features)
    return meta_examples
