"""Measure the gap between two models."""
from __future__ import annotations

import copy

from explainaboard.analysis.analyses import AnalysisResult, BucketAnalysisResult
from explainaboard.analysis.performance import BucketPerformance, Performance
from explainaboard.analysis.result import Result
from explainaboard.info import SysOutputInfo


def _diff_overall(
    sys1: dict[str, Performance], sys2: dict[str, Performance]
) -> dict[str, Performance]:
    """Helper function to make a difference performance.

    Args:
        sys1: System1 overall performances.
        sys2: System1 overall performances.

    Returns:
        dict of Perforamnces generated from sys1 and sys2.
    """
    return {k: Performance(sys1[k].value - sys2[k].value) for k in sys1}


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
    overall = [
        _diff_overall(o1, o2)
        for o1, o2 in zip(sys1.results.overall, sys2.results.overall)
    ]

    analyses: list[AnalysisResult] = []

    for analysis1, analysis2 in zip(sys1.results.analyses, sys2.results.analyses):
        if not isinstance(analysis1, BucketAnalysisResult):
            analyses.append(copy.deepcopy(analysis1))
            continue

        if not isinstance(analysis2, BucketAnalysisResult):
            raise ValueError(
                "Mismatched analyses: "
                f"{type(analysis1).__name__} v.s. {type(analysis2).__name__}"
            )

        analyses.append(
            BucketAnalysisResult(
                name=analysis1.name,
                level=analysis1.level,
                bucket_performances=[
                    BucketPerformance(
                        n_samples=bp1.n_samples,
                        bucket_samples=bp1.bucket_samples[:],
                        performances=_diff_overall(bp1.performances, bp2.performances),
                        bucket_interval=bp1.bucket_interval,
                        bucket_name=bp1.bucket_name,
                    )
                    for bp1, bp2 in zip(
                        analysis1.bucket_performances, analysis2.bucket_performances
                    )
                ],
            )
        )

    sys = copy.deepcopy(sys1)
    sys.results = Result(overall=overall, analyses=analyses)
    return sys
