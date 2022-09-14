"""Measure the gap between two models."""

import copy

from explainaboard.analysis.analyses import BucketAnalysisResult
from explainaboard.info import SysOutputInfo
from explainaboard.utils.typing_utils import unwrap


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
    sys = copy.deepcopy(sys1)

    orm, or1, or2 = (unwrap(x.results.overall) for x in (sys, sys1, sys2))
    for orm_lev, or1_lev, or2_lev in zip(orm, or1, or2):
        for orm_met, or1_met, or2_met in zip(orm_lev, or1_lev, or2_lev):
            orm_met.value = float(or1_met.value) - float(or2_met.value)
            orm_met.confidence_score_low = None
            orm_met.confidence_score_high = None

    fgr, fgr1, fgr2 = (unwrap(x.results.analyses) for x in (sys, sys1, sys2))
    for fgr_buks, fgr1_buks, fgr2_buks in zip(fgr, fgr1, fgr2):
        if (
            not isinstance(fgr_buks, BucketAnalysisResult)
            or not isinstance(fgr1_buks, BucketAnalysisResult)
            or not isinstance(fgr2_buks, BucketAnalysisResult)
        ):
            continue
        for fgr_buk, fgr1_buk, fgr2_buk in zip(
            fgr_buks.bucket_performances,
            fgr1_buks.bucket_performances,
            fgr2_buks.bucket_performances,
        ):
            for fgr_met, fgr1_met, fgr2_met in zip(
                fgr_buk.performances, fgr1_buk.performances, fgr2_buk.performances
            ):
                fgr_met.value = fgr1_met.value - fgr2_met.value
                # TODO(gneubig): these could be done via pairwise bootstraps
                fgr_met.confidence_score_low = None
                fgr_met.confidence_score_high = None

    return sys
