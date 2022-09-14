"""Base classes to specify analyses."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, final, Optional, TypeVar

import numpy as np

import explainaboard.analysis.bucketing
from explainaboard.analysis.case import AnalysisCase, AnalysisCaseCollection
from explainaboard.analysis.feature import FeatureType, get_feature_type_serializer
from explainaboard.analysis.performance import BucketPerformance, Performance
from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import metric_config_from_dict
from explainaboard.utils.typing_utils import narrow, unwrap, unwrap_generator


# See https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AnalysisResult(metaclass=abc.ABCMeta):
    """A base class specifying the result of an analysis.

    The actual details of the result will be implemented by the inheriting class.

    Attributes:
        name: The name of the analysis.
        level: The level that the analysis belongs to.
    """

    name: str
    level: str

    @abc.abstractmethod
    def generate_report(self) -> str:
        """Generate human-readable report.

        Returns:
            Multi-lined string representing the report of this result.
        """
        ...

    @staticmethod
    def from_dict(dikt):
        """Deserialization method."""
        type = dikt.pop('cls_name')
        if type == 'BucketAnalysisResult':
            return BucketAnalysisResult.from_dict(dikt)
        elif type == 'ComboCountAnalysisResult':
            return ComboCountAnalysisResult.from_dict(dikt)
        else:
            raise ValueError(f'bad AnalysisResult type {type}')


@dataclass
class Analysis:
    """A super-class for analyses.

    Analyses take in examples and analyze their features in
    some way. The exact analysis performed will vary depending on the inheriting
    class.

    Attributes:
        description: a textual description of the analysis to be performed
        level: The level that the analysis belongs to.
    """

    description: str | None
    level: str

    def perform(
        self,
        cases: list[AnalysisCase],
        metrics: list[Metric],
        stats: list[MetricStats],
        confidence_alpha: float,
    ) -> AnalysisResult:
        """Perform the analysis.

        Args:
            cases: The list of analysis cases over which to perform the analysis.
              These could be examples, spans, tokens, etc.
            metrics: The metrics used to evaluate the cases.
            stats: The statistics calculated by each metric.
            confidence_alpha: In the case that any significance analysis is performed,
                      the inverse confidence level.

        Returns:
            The result of the analysis.
        """
        raise NotImplementedError

    @staticmethod
    def from_dict(dikt: dict):
        """Deserialization method."""
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
                features=tuple(dikt['features']),
            )


@final
@dataclass
class BucketAnalysisResult(AnalysisResult):
    """A result of running a `BucketAnalysis`.

    Attributes:
        bucket_performances: A list of performances bucket-by-bucket, including the
          interval over which the bucket is calculated, the performance itself, etc.
          See `BucketPerformance` for more details.
        cls_name: The name of the class.
    """

    bucket_performances: list[BucketPerformance]
    cls_name: Optional[str] = None

    @staticmethod
    def from_dict(dikt: dict) -> BucketAnalysisResult:
        """Deserialization method."""
        bucket_performances = [
            BucketPerformance.from_dict(v1) for v1 in dikt['bucket_performances']
        ]
        return BucketAnalysisResult(
            name=dikt['name'],
            level=dikt['level'],
            bucket_performances=bucket_performances,
        )

    def __post_init__(self):
        """Set the class name and validate."""
        metric_names = [x.metric_name for x in self.bucket_performances[0].performances]
        num_metrics = len(metric_names)
        for bucket_perf in self.bucket_performances:
            if len(bucket_perf.performances) != num_metrics:
                raise ValueError(
                    "Inconsistent number of metrics. "
                    f"Required: {num_metrics}, got: {len(bucket_perf.performances)}"
                )
            for metric_name, perf in zip(metric_names, bucket_perf.performances):
                if perf.metric_name != metric_name:
                    raise ValueError(
                        "Inconsistent metric names. "
                        f"Required: {metric_name}, got: {perf.metric_name}"
                    )

        self.cls_name: str = self.__class__.__name__

    def generate_report(self) -> str:
        """See AnalysisResult.generate_report."""
        texts: list[str] = []

        metric_names = [x.metric_name for x in self.bucket_performances[0].performances]

        for i, metric_name in enumerate(metric_names):
            texts.append(f"the information of #{self.name}#")
            texts.append(f"bucket_name\t{metric_name}\t#samples")

            for bucket_perf in self.bucket_performances:
                perf = bucket_perf.performances[i]

                if bucket_perf.bucket_interval is not None:
                    bucket_name = f"{unwrap(bucket_perf.bucket_interval)}"
                else:
                    bucket_name = unwrap(bucket_perf.bucket_name)

                texts.append(
                    f"{bucket_name}\t" f"{perf.value}\t" f"{bucket_perf.n_samples}"
                )

            texts.append('')

        return "\n".join(texts)


@final
@dataclass
class BucketAnalysis(Analysis):
    """Perform an analysis of various examples bucketed by features.

    Depending on which `method` is chosen here, the way bucketing is performed will be
    different. See the documentation of each function in the
    `explainaboard.analysis.bucketing` package for more details.

    Attributes:
        feature: the name of the feature to bucket
        method: the bucket strategy, can be "continuous", "discrete", or "fixed"
        number: the number of buckets to be used
        setting: parameters of bucketing, varying by `method`
        sample_limit: an upper limit on the number of samples saved in each bucket.
        cls_name: the name of the class.
    """

    feature: str
    method: str = "continuous"
    number: int = 4
    setting: Any = None  # For different bucket_methods, the settings are diverse
    sample_limit: int = 50
    cls_name: Optional[str] = None

    def __post_init__(self):
        """Set the class name."""
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
        confidence_alpha: float,
    ) -> AnalysisResult:
        """See Analysis.perform."""
        # Preparation for bucketing
        bucket_func: Callable[..., list[AnalysisCaseCollection]] = getattr(
            explainaboard.analysis.bucketing,
            self.method,
        )

        if len(cases) == 0 or self.feature not in cases[0].features:
            raise RuntimeError(f"bucket analysis: feature {self.feature} not found.")

        samples_over_bucket = bucket_func(
            sample_features=[(x, x.features[self.feature]) for x in cases],
            bucket_number=self.number,
            bucket_setting=self.setting,
        )

        bucket_performances: list[BucketPerformance] = []
        for bucket_collection in samples_over_bucket:
            # Subsample examples to save
            subsampled_ids = self._subsample_analysis_cases(bucket_collection.samples)

            n_samples = len(bucket_collection.samples)
            bucket_performance = BucketPerformance(
                n_samples=n_samples,
                bucket_samples=subsampled_ids,
                bucket_interval=bucket_collection.interval,
                bucket_name=bucket_collection.name,
            )

            for metric_func, metric_stat in zip(
                unwrap_generator(metrics),
                unwrap_generator(stats),
            ):
                # Samples may be empty when user defined a bucket interval that
                # has no samples
                if n_samples == 0.0:
                    value = 0.0
                    conf_low: Optional[float] = None
                    conf_high: Optional[float] = None
                else:
                    bucket_stats = metric_stat.filter(bucket_collection.samples)
                    metric_result = metric_func.evaluate_from_stats(
                        bucket_stats,
                        confidence_alpha=confidence_alpha,
                    )

                    conf_low, conf_high = (
                        metric_result.confidence_interval
                        if metric_result.confidence_interval
                        else (None, None)
                    )

                    value = metric_result.value

                performance = Performance(
                    metric_name=metric_func.config.name,
                    value=value,
                    confidence_score_low=conf_low,
                    confidence_score_high=conf_high,
                )

                bucket_performance.performances.append(performance)

            bucket_performances.append(bucket_performance)

        return BucketAnalysisResult(
            name=self.feature, level=self.level, bucket_performances=bucket_performances
        )


@final
@dataclass
class ComboCountAnalysisResult(AnalysisResult):
    """A result of running a `ComboCountAnalysis`.

    Attributes:
        features: A tuple of strings, representing the feature names that were
          analyzed
        combo_counts: A list of tuples. The first tuple element is the feature
          values corresponding to the feature names in `features`. The second element is
          the count of that feature combination in the corpus.
    """

    features: tuple[str, ...]
    combo_counts: list[tuple[tuple[str, ...], int]]
    cls_name: Optional[str] = None

    @staticmethod
    def from_dict(dikt: dict) -> ComboCountAnalysisResult:
        """Deserialization method."""
        return ComboCountAnalysisResult(
            name=dikt['name'],
            level=dikt['level'],
            features=dikt['features'],
            combo_counts=dikt['combo_counts'],
        )

    def __post_init__(self):
        """Set the class name and validate."""
        num_features = len(self.features)
        for k, _ in self.combo_counts:
            if len(k) != num_features:
                raise ValueError(
                    "Inconsistent number of features. "
                    f"Required: {num_features}, got: {len(k)}"
                )

        self.cls_name: str = self.__class__.__name__

    def generate_report(self) -> str:
        """See AnalysisResult.generate_report."""
        texts: list[str] = []

        texts.append('feature combos for ' + ', '.join(self.features))
        texts.append('\t'.join(self.features + ('#',)))

        for k, v in sorted(self.combo_counts):
            texts.append('\t'.join(k + (str(v),)))

        texts.append('')
        return "\n".join(texts)


@final
@dataclass
class ComboCountAnalysis(Analysis):
    """A class used to count feature combinations (e.g. for confusion matrices).

    It will return counts of each combination of values for the features named in
    `features`.

    Args:
        features: the name of the features over which to perform the analysis
        cls_name: the name of the class
    """

    features: tuple[str, ...]
    cls_name: Optional[str] = None

    def __post_init__(self):
        """Set the class name."""
        self.cls_name: str = self.__class__.__name__

    AnalysisCaseType = TypeVar('AnalysisCaseType')

    def perform(
        self,
        cases: list[AnalysisCase],
        metrics: list[Metric],
        stats: list[MetricStats],
        confidence_alpha: float,
    ) -> AnalysisResult:
        """See Analysis.perform."""
        for x in self.features:
            if x not in cases[0].features:
                raise RuntimeError(f"combo analysis: feature {x} not found.")

        combo_map: dict[tuple[str, ...], int] = {}
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
    """Specifies the features of a particular level at which analysis is performed.

    Args:
        name: the name of the analysis level
        features: the features specified for the analysis level
        metric_configs: configurations of the metrics to be applied to the level
    """

    name: str
    features: dict[str, FeatureType]
    metric_configs: list[MetricConfig]

    @staticmethod
    def from_dict(dikt: dict):
        """Deserialization method."""
        ft_serializer = get_feature_type_serializer()

        features = {
            # See https://github.com/python/mypy/issues/4717
            k: narrow(FeatureType, ft_serializer.deserialize(v))  # type: ignore
            for k, v in dikt['features'].items()
        }
        metric_configs = [metric_config_from_dict(v) for v in dikt['metric_configs']]
        return AnalysisLevel(
            name=dikt['name'],
            features=features,
            metric_configs=metric_configs,
        )
