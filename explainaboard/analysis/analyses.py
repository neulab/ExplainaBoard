from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, TypeVar

import numpy as np

import explainaboard.analysis.bucketing
from explainaboard.analysis.case import AnalysisCase, AnalysisCaseCollection
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.performance import BucketPerformance, Performance
from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import metric_config_from_dict
from explainaboard.utils.logging import get_logger
from explainaboard.utils.typing_utils import unwrap_generator


@dataclass
class AnalysisResult:
    """
    A result of an analysis, where the actual details of the result will be implemented
    by the inheriting class.
    :param name: The name of the analysis.
    """

    name: str
    level: str

    def print(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(dikt):
        type = dikt.pop('cls_name')
        if type == 'BucketAnalysisResult':
            return BucketAnalysisResult.from_dict(dikt)
        elif type == 'ComboCountAnalysisResult':
            return ComboCountAnalysisResult.from_dict(dikt)
        else:
            raise ValueError(f'bad AnalysisResult type {type}')


@dataclass
class Analysis:
    """
    A super-class for analyses, which take in examples and analyze their features in
    some way. The exact analysis performed will vary depending on the inheriting
    class.
    :param description: a textual description of the analysis to be performed
    """

    description: str | None
    level: str

    def perform(
        self,
        cases: list[AnalysisCase],
        metrics: list[Metric],
        stats: list[MetricStats],
        conf_value: float,
    ) -> AnalysisResult | None:
        """
        A super-class for analyses, which take in examples and analyze their features in
        some way. The exact analysis performed will vary depending on the inheriting
        class.
        :param cases: The list of analysis cases over which to perform the analysis.
          These could be examples, spans, tokens, etc.
        :param metrics: The metrics used to evaluate the cases.
        :param stats: The statistics calculated by each metric.
        :conf_value: In the case that any significance analysis is performed, the
          confidence level.
        """
        raise NotImplementedError

    @staticmethod
    def from_dict(dikt: dict):
        type = dikt.pop('cls_name')
        if type == 'BucketAnalysis':
            return BucketAnalysis(
                description=dikt.get('description'),
                level=dikt['level'],
                feature=dikt['feature'],
                method=dikt.get('method', 'continuous'),
                number=dikt.get('number', 4),
                setting=dikt.get('setting'),
                sample_limit=dikt.get('sample_limit', 50),
            )
        elif type == 'ComboCountAnalysis':
            return ComboCountAnalysis(
                description=dikt.get('description'),
                level=dikt['level'],
                features=dikt['features'],
            )


@dataclass
class BucketAnalysisResult(AnalysisResult):
    """
    A result of running a `BucketAnalysis`.
    :param bucket_performances: A list of performances bucket-by-bucket, including the
      interval over which the bucket is calculated, the performance itself, etc.
      See `BucketPerformance` for more details.
    """

    bucket_performances: list[BucketPerformance]
    cls_name: Optional[str] = None

    @staticmethod
    def from_dict(dikt: dict) -> BucketAnalysisResult:
        bucket_performances = [
            BucketPerformance.from_dict(v1) for v1 in dikt['bucket_performances']
        ]
        return BucketAnalysisResult(
            name=dikt['name'],
            level=dikt['level'],
            bucket_performances=bucket_performances,
        )

    def __post_init__(self):
        self.cls_name: str = self.__class__.__name__

    def print(self):
        metric_names = [x.metric_name for x in self.bucket_performances[0].performances]
        for i, metric_name in enumerate(metric_names):
            get_logger('report').info(f"the information of #{self.name}#")
            get_logger('report').info(f"bucket_interval\t{metric_name}\t#samples")
            for bucket_perf in self.bucket_performances:
                get_logger('report').info(
                    f"{bucket_perf.bucket_interval}\t"
                    f"{bucket_perf.performances[i].value}\t"
                    f"{bucket_perf.n_samples}"
                )
            get_logger('report').info('')


@dataclass
class BucketAnalysis(Analysis):
    """
    Perform an analysis of various examples bucketed by features.
    Depending on which `method` is chosen here, the way bucketing is performed will be
    different. See the documentation of each function in the
    `explainaboard.analysis.bucketing` package for more details.
    Args:
        feature: the name of the feature to bucket
        method: the bucket strategy, can be "continuous", "discrete", or "fixed"
        number: the number of buckets to be used
        setting: parameters of bucketing, varying by `method`
    """

    feature: str
    method: str = "continuous"
    number: int = 4
    setting: Any = None  # For different bucket_methods, the settings are diverse
    sample_limit: int = 50
    cls_name: Optional[str] = None

    def __post_init__(self):
        self.cls_name: str = self.__class__.__name__

    AnalysisCaseType = TypeVar('AnalysisCaseType')

    def _subsample_analysis_cases(self, analysis_cases: list[int]) -> list[int]:
        if len(analysis_cases) > self.sample_limit:
            sample_ids = np.random.choice(
                analysis_cases, self.sample_limit, replace=False
            )
            return [int(x) for x in sample_ids]
        else:
            return analysis_cases

    def perform(
        self,
        cases: list[AnalysisCase],
        metrics: list[Metric],
        stats: list[MetricStats],
        conf_value: float,
    ) -> AnalysisResult | None:
        # Preparation for bucketing
        bucket_func: Callable[..., list[AnalysisCaseCollection]] = getattr(
            explainaboard.analysis.bucketing,
            self.method,
        )
        if len(cases) == 0 or self.feature not in cases[0].features:
            get_logger().warning(
                f'bucket analysis: feature {self.feature} not found, ' f'skipping'
            )
            return None
        samples_over_bucket = bucket_func(
            sample_features=[(x, x.features[self.feature]) for x in cases],
            bucket_number=self.number,
            bucket_setting=self.setting,
        )

        bucket_performances: list[BucketPerformance] = []
        for bucket_collection in samples_over_bucket:
            # Subsample examples to save
            subsampled_ids = self._subsample_analysis_cases(bucket_collection.samples)

            bucket_performance = BucketPerformance(
                bucket_interval=bucket_collection.interval,
                n_samples=len(bucket_collection.samples),
                bucket_samples=subsampled_ids,
            )

            for metric_func, metric_stat in zip(
                unwrap_generator(metrics),
                unwrap_generator(stats),
            ):
                bucket_stats = metric_stat.filter(bucket_collection.samples)
                metric_result = metric_func.evaluate_from_stats(
                    bucket_stats,
                    conf_value=conf_value,
                )

                conf_low, conf_high = (
                    metric_result.conf_interval
                    if metric_result.conf_interval
                    else (None, None)
                )

                performance = Performance(
                    metric_name=metric_func.config.name,
                    value=metric_result.value,
                    confidence_score_low=conf_low,
                    confidence_score_high=conf_high,
                )

                bucket_performance.performances.append(performance)

            bucket_performances.append(bucket_performance)

        return BucketAnalysisResult(
            name=self.feature, level=self.level, bucket_performances=bucket_performances
        )


@dataclass
class ComboCountAnalysisResult(AnalysisResult):
    """
    A result of running a `ComboCountAnalysis`.
    :param features: A tuple of strings, representing the feature names that were
      analyzed
    :param combo_counts: A list of tuples. The first tuple element is the feature
      values corresponding to the feature names in `features`. The second element is
      the count of that feature combination in the corpus.
    """

    features: tuple
    combo_counts: list[tuple[tuple, int]] | None = None
    cls_name: Optional[str] = None

    @staticmethod
    def from_dict(dikt: dict) -> ComboCountAnalysisResult:
        return ComboCountAnalysisResult(
            name=dikt['name'],
            level=dikt['level'],
            features=dikt['features'],
            combo_counts=dikt['combo_counts'],
        )

    def __post_init__(self):
        self.cls_name: str = self.__class__.__name__

    def print(self):
        get_logger('report').info('feature combos for ' + ', '.join(self.features))
        get_logger('report').info('\t'.join(self.features + ('#',)))
        for k, v in sorted(self.combo_counts):
            get_logger('report').info('\t'.join(k + (str(v),)))
        get_logger('report').info('')


@dataclass
class ComboCountAnalysis(Analysis):
    """
    A class used to count feature combinations (e.g. for confusion matrices). It will
    return counts of each combination of values for the features named in `features`.
    Args:
        level: the level to which this analysis should be applied
        features: the name of the features over which to perform the analysis
    """

    features: tuple
    cls_name: Optional[str] = None

    def __post_init__(self):
        self.cls_name: str = self.__class__.__name__

    AnalysisCaseType = TypeVar('AnalysisCaseType')

    def perform(
        self,
        cases: list[AnalysisCase],
        metrics: list[Metric],
        stats: list[MetricStats],
        conf_value: float,
    ) -> AnalysisResult | None:
        if len(cases) == 0:
            return None
        for x in self.features:
            if x not in cases[0].features:
                get_logger().warning(
                    f'combo analysis: feature {x} not found, ' f'skipping'
                )
                return None
        combo_map: dict[tuple, int] = {}
        for case in cases:
            feat_vals = tuple([case.features[x] for x in self.features])
            combo_map[feat_vals] = combo_map.get(feat_vals, 0) + 1
        combo_list = list(combo_map.items())
        return ComboCountAnalysisResult(
            name='combo(' + ','.join(self.features) + ')',
            level=self.level,
            features=self.features,
            combo_counts=combo_list,
        )


@dataclass
class AnalysisLevel:
    name: str
    features: dict[str, FeatureType]
    metric_configs: list[MetricConfig]

    @staticmethod
    def from_dict(dikt: dict):
        features = {k: FeatureType.from_dict(v) for k, v in dikt['features'].items()}
        metric_configs = [metric_config_from_dict(v) for v in dikt['metric_configs']]
        return AnalysisLevel(
            name=dikt['name'],
            features=features,
            metric_configs=metric_configs,
        )
