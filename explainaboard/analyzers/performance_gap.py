import copy

from explainaboard.info import SysOutputInfo
from explainaboard.utils.typing_utils import unwrap


def get_pairwise_performance_gap(
    sys1: SysOutputInfo, sys2: SysOutputInfo
) -> SysOutputInfo:

    sys = copy.deepcopy(sys1)

    orm, or1, or2 = (unwrap(x.results.overall) for x in (sys, sys1, sys2))
    for metric_name, performance_unit in orm.items():
        orm[metric_name].value = float(or1[metric_name].value) - float(
            or2[metric_name].value
        )
        orm[metric_name].confidence_score_low = None
        orm[metric_name].confidence_score_high = None

    fgr, fgr1, fgr2 = (unwrap(x.results.fine_grained) for x in (sys, sys1, sys2))
    for bucket_attr, buckets in fgr.items():
        for bucket_name, bucket in buckets.items():
            for perf_id, perf in enumerate(bucket.performances):
                perf.value = float(
                    fgr1[bucket_attr][bucket_name].performances[perf_id].value
                ) - float(fgr2[bucket_attr][bucket_name].performances[perf_id].value)
                # TODO(gneubig): these could be done via pairwise bootstraps
                perf.confidence_score_low = None
                perf.confidence_score_high = None

    return sys
