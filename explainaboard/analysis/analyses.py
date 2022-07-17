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
    name: str

    def print(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(dikt):
        type = dikt.pop('_type')
        if type == 'BucketAnalysisResult':
            bucket_performances = [
                BucketPerformance.from_dict(v1) for v1 in dikt['bucket_performances']
            ]
            return BucketAnalysisResult(
                name=dikt['name'], bucket_performances=bucket_performances
            )
        else:
            raise ValueError(f'bad AnalysisResult type {type}')


@dataclass
class Analysis:
    def perform(
        self,
        cases: list[AnalysisCase],
        metrics: list[Metric],
        stats: list[MetricStats],
        conf_value: float,
    ) -> AnalysisResult | None:
        raise NotImplementedError

    @staticmethod
    def from_dict(dikt: dict):
        type = dikt.pop('_type')
        if type == 'BucketAnalysis':
            return BucketAnalysis(
                feature=dikt['feature'],
                method=dikt.get('method', 'continuous'),
                number=dikt.get('number', 4),
                setting=dikt.get('setting'),
                sample_limit=dikt.get('sample_limit', 50),
            )
        else:
            raise ValueError(f'invalid type {type}')


@dataclass
class BucketAnalysisResult(AnalysisResult):
    bucket_performances: list[BucketPerformance]
    _type: Optional[str] = None

    def __post_init__(self):
        self._type: str = self.__class__.__name__

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
    The class is used to define a dataclass for bucketing strategy
    Args:
        feature: the name of the feature to bucket
        method: the bucket strategy
        number: the number of buckets to be bucketed
        setting: parameters of bucketing
    """

    feature: str
    method: str = "continuous"
    number: int = 4
    setting: Any = None  # For different bucket_methods, the settings are diverse
    sample_limit: int = 50
    _type: Optional[str] = None

    def __post_init__(self):
        self._type: str = self.__class__.__name__

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

        return BucketAnalysisResult(self.feature, bucket_performances)

@dataclass
class ComboCountAnalysisResult(AnalysisResult):
    features: tuple
    combo_counts: dict[tuple, int] | None = None
    _type: Optional[str] = None

    def __post_init__(self):
        self._type: str = self.__class__.__name__

    def print(self):
        get_logger('report').info('feature combos for '+', '.join(self.features))
        get_logger('report').info('\t'.join(self.features + ('#',)))
        for k in sorted(self.combo_counts.keys()):
            get_logger('report').info('\t'.join(k + (str(self.combo_counts[k]),)))
        get_logger('report').info('')

    def json_repr(self):
        sanitized_counts: dict[str,int] = {}
        for k, v in self.combo_counts.items():
        return {'features': '\t'.join(self.features), 'combo_counts': sanitized_counts}


@dataclass
class ComboCountAnalysis(Analysis):
    """
    A class used to count feature combinations (e.g. for confusion matrices)
    Args:
        features: the name of the feature to bucket
    """

    features: tuple
    _type: Optional[str] = None

    def __post_init__(self):
        self._type: str = self.__class__.__name__

    AnalysisCaseType = TypeVar('AnalysisCaseType')

    def perform(
            self,
            cases: list[AnalysisCase],
            metrics: list[Metric],
            stats: list[MetricStats],
            conf_value: float,
    ) -> AnalysisResult | None:
        if len(cases) == 0 or any((x not in cases[0].features) for x in self.features):
            return None
        combo_counts: dict[tuple, int] = {}
        for case in cases:
            feat_vals = tuple([case.features[x] for x in self.features])
            combo_counts[feat_vals] = combo_counts.get(feat_vals, 0) + 1
        return ComboCountAnalysisResult(
            name='combo('+','.join(self.features)+')',
            features=self.features,
            combo_counts=combo_counts,
        )


@dataclass
class AnalysisLevel:
    name: str
    features: dict[str, FeatureType]
    analyses: list[Analysis]
    metric_configs: list[MetricConfig]

    @staticmethod
    def from_dict(dikt: dict):
        features = {k: FeatureType.from_dict(v) for k, v in dikt['features'].items()}
        analyses = [Analysis.from_dict(v) for v in dikt['analyses']]
        metric_configs = [metric_config_from_dict(v) for v in dikt['metric_configs']]
        return AnalysisLevel(
            name=dikt['name'],
            features=features,
            analyses=analyses,
            metric_configs=metric_configs,
        )
