from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Callable, TypeVar

import numpy as np

from explainaboard.analysis.case import AnalysisCase, AnalysisCaseCollection
from explainaboard.info import BucketPerformance, SysOutputInfo, \
    Performance
import explainaboard.analysis.bucketing
from explainaboard.metrics.metric import MetricStats, Metric
from explainaboard.utils.typing_utils import unwrap_generator


@dataclass
class AnalysisResult:
    name: str


@dataclass
class Analysis:
    def perform(self, sys_info: SysOutputInfo, cases: list[AnalysisCase], metrics: list[Metric], stats: list[MetricStats]) -> AnalysisResult:
        raise NotImplementedError


@dataclass
class BucketAnalysisResult(AnalysisResult):
    bucket_performances: list[BucketPerformance]



@dataclass
class BucketAnalysis(Analysis):
    """
    The class is used to define a dataclass for bucketing strategy
    Args:
        feature: the name of the feature to bucket
        method: the bucket strategy
        number: the number of buckets to be bucketed
        setting: hyper-paraterms of bucketing
    """

    feature: str
    method: str = "continuous"
    number: int = 4
    setting: Any = 1  # For different bucket_methods, the settings are diverse
    sample_limit: int = 50
    _type: Optional[str] = None

    def __post_init__(self):
        self._type: str = self.__class__.__name__

    AnalysisCaseType = TypeVar('AnalysisCaseType')

    def _subsample_analysis_cases(
        self, analysis_cases: list[AnalysisCaseType]
    ) -> list[AnalysisCaseType]:
        if len(analysis_cases) > self.sample_limit:
            ids = np.array(range(len(analysis_cases)))
            bucket_sample_ids = np.random.choice(
                ids, self.sample_limit, replace=False
            )
            return [analysis_cases[i] for i in bucket_sample_ids]
        else:
            return analysis_cases

    def perform(self, sys_info: SysOutputInfo, cases: list[AnalysisCase], metrics: list[Metric], stats: list[MetricStats]) -> AnalysisResult:
        # Preparation for bucketing
        bucket_func: Callable[..., list[AnalysisCaseCollection]] = getattr(
            explainaboard.analysis.bucketing,
            self.method,
        )
        samples_over_bucket = bucket_func(
            sample_features=[
                (x, x.features[self.feature])
                for x in cases
            ],
            bucket_number=self.number,
            bucket_setting=self.setting,
        )

        bucket_performances: list[BucketPerformance] = []
        for bucket_collection in samples_over_bucket:
            analysis_cases = bucket_collection.samples
            sample_ids = [analysis_case.sample_id for analysis_case in analysis_cases]

            # Subsample examples to save
            bucket_samples = self._subsample_analysis_cases(analysis_cases)

            bucket_performance = BucketPerformance(
                bucket_interval=bucket_collection.interval,
                n_samples=len(analysis_cases),
                bucket_samples=bucket_samples,
            )

            for metric_cfg, metric_func, metric_stat in zip(
                    unwrap_generator(sys_info.metric_configs),
                    unwrap_generator(metrics),
                    unwrap_generator(stats),
            ):
                bucket_stats = metric_stat.filter(sample_ids)
                metric_result = metric_func.evaluate_from_stats(
                    bucket_stats,
                    conf_value=sys_info.conf_value,
                )

                conf_low, conf_high = (
                    metric_result.conf_interval
                    if metric_result.conf_interval
                    else (None, None)
                )

                performance = Performance(
                    metric_name=metric_cfg.name,
                    value=metric_result.value,
                    confidence_score_low=conf_low,
                    confidence_score_high=conf_high,
                )

                bucket_performance.performances.append(performance)

            bucket_performances.append(bucket_performance)

        return BucketAnalysisResult(self.feature, bucket_performances)



