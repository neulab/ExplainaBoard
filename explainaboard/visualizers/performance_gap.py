"""Measure the gap between two models."""
from __future__ import annotations

import copy

from explainaboard.analysis.analyses import AnalysisResult, BucketAnalysisDetails
from explainaboard.analysis.performance import BucketPerformance
from explainaboard.analysis.result import Result
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.metric import MetricResult, Score


def _diff_overall(
    sys1: dict[str, MetricResult], sys2: dict[str, MetricResult]
) -> dict[str, MetricResult]:
    """Helper function to make a difference performance.

    Args:
        sys1: System1 overall performances.
        sys2: System1 overall performances.

    Returns:
        dict of Perforamnces generated from sys1 and sys2.
    """
    return {
        k: MetricResult(
            {
                "score": Score(
                    sys1[k].get_value(Score, "score").value
                    - sys2[k].get_value(Score, "score").value
                )
            }
        )
        for k in sys1
    }


def get_pairwise_performance_gap(
    sys1: SysOutputInfo, sys2: SysOutputInfo
) -> SysOutputInfo:
    """Measure the performance gap between two models.

    Args:
        sys1: Information from the first system.
        sys2: Information from the second system.

    Returns:
        A SystemOutputInfo object that has the difference between the performances.
    """
    overall = {
        level: _diff_overall(sys1.results.overall[level], sys2.results.overall[level])
        for level in sys1.results.overall
    }

    analyses: list[AnalysisResult] = []

    for analysis1, analysis2 in zip(sys1.results.analyses, sys2.results.analyses):
        if not isinstance(analysis1.details, BucketAnalysisDetails):
            analyses.append(copy.deepcopy(analysis1))
            continue

        if not isinstance(analysis2.details, BucketAnalysisDetails):
            raise ValueError(
                "Mismatched analyses: "
                f"{type(analysis1).__name__} v.s. {type(analysis2).__name__}"
            )

        analyses.append(
            AnalysisResult(
                name=analysis1.name,
                level=analysis1.level,
                details=BucketAnalysisDetails(
                    bucket_performances=[
                        BucketPerformance(
                            n_samples=bp1.n_samples,
                            bucket_samples=bp1.bucket_samples[:],
                            results=_diff_overall(bp1.results, bp2.results),
                            bucket_interval=bp1.bucket_interval,
                            bucket_name=bp1.bucket_name,
                        )
                        for bp1, bp2 in zip(
                            analysis1.details.bucket_performances,
                            analysis2.details.bucket_performances,
                        )
                    ],
                ),
            )
        )

    sys = copy.deepcopy(sys1)
    sys.results = Result(overall=overall, analyses=analyses)
    return sys
