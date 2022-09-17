"""A parent class to represent processors."""

from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Any, final, Optional

from eaas.async_client import AsyncClient
from eaas.config import Config

from explainaboard import TaskType
from explainaboard.analysis.analyses import (
    Analysis,
    AnalysisLevel,
    AnalysisResult,
    BucketAnalysis,
    BucketAnalysisResult,
)
from explainaboard.analysis.case import AnalysisCase
from explainaboard.analysis.feature import (
    DataType,
    FeatureType,
    get_feature_type_serializer,
    Value,
)
from explainaboard.analysis.performance import BucketPerformance, Performance
from explainaboard.analysis.result import Result
from explainaboard.info import OverallStatistics, SysOutputInfo
from explainaboard.loaders import DatalabLoaderOption, get_loader_class
from explainaboard.metrics.metric import MetricConfig, MetricStats
from explainaboard.utils.cache_api import (
    read_statistics_from_cache,
    write_statistics_to_cache,
)
from explainaboard.utils.logging import get_logger, progress
from explainaboard.utils.tokenizer import get_default_tokenizer, Tokenizer
from explainaboard.utils.typing_utils import narrow, unwrap, unwrap_generator


class Processor(metaclass=abc.ABCMeta):
    """Base case for task-based processor."""

    @classmethod
    @abc.abstractmethod
    def task_type(cls) -> TaskType:
        """Returns the task type of this processor."""
        ...

    def get_tokenizer(self, lang: str | None) -> Tokenizer:
        """Return a tokenizer based on the language.

        Args:
            lang: the name of a language code.

        Returns:
            A suitable tokenizer for the specified language.
        """
        return get_default_tokenizer(lang)

    @abc.abstractmethod
    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """Returns a list of analysis levels.

        These indicate different views of the higher-level example. For instance, a task
        may perform 'example'-level analysis, and 'token'-level analysis in which case
        this list would have one level for each.
        """
        ...

    @abc.abstractmethod
    def default_analyses(self) -> list[Analysis]:
        """Returns the analyses to be performed."""
        ...

    def continuous_feature_analyses(self) -> list[Analysis]:
        """Return analyses over all continuous features in the analysis levels."""
        analyses: list[Analysis] = []
        analysis_levels = self.default_analysis_levels()
        for lev in analysis_levels:
            # Continuous features
            for k, v in lev.features.items():
                if isinstance(v, Value) and v.dtype == DataType.FLOAT:
                    analyses.append(
                        BucketAnalysis(
                            level=lev.name,
                            description=lev.features[k].description,
                            feature=k,
                            method="continuous",
                        )
                    )
        return analyses

    @classmethod
    @abc.abstractmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        """Returns the default metrics of this processor."""
        ...

    @classmethod
    def full_metric_list(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        """Returns an extensive list of metrics that may be used."""
        return cls.default_metrics(
            level=level,
            source_language=source_language,
            target_language=target_language,
        )

    @classmethod
    def metric_is_valid(cls, metric_config: MetricConfig) -> bool:
        """Checks if a particular metric is valid for a particular task."""
        return True

    def __init__(self) -> None:
        """Constructor."""
        # Things to use only if necessary
        self._eaas_config: Optional[Config] = None
        self._eaas_client: Optional[AsyncClient] = None
        # self._statistics_func = None
        self._preprocessor = None
        # A limit on the number of samples stored for each bucket. Hard-coded for now
        self._bucket_sample_limit = 50

    def _get_statistics_resources(self, sys_info: SysOutputInfo) -> dict[str, Any]:
        """From a DataLab dataset split, get resources to calculate statistics."""
        return {"cls": self, "sys_info": sys_info}  #

    @abc.abstractmethod
    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        ...

    def _gen_external_stats(self, sys_info: SysOutputInfo):
        """Generate external statistics.

        These are gathered from a relatively costly source, such as the training set,
        then cached for future use.

        Args:
            sys_info: Information about the system outputs

        Returns:
            Statistics from, usually, the training set that are used to calculate
            other features
        """
        statistics = None
        if sys_info.dataset_name is not None:
            split_name = "train"
            sub_dataset = (
                None
                if sys_info.sub_dataset_name == "default"
                else sys_info.sub_dataset_name
            )
            # read statistics from cache
            if sys_info.reload_stat:
                statistics = read_statistics_from_cache(
                    sys_info.dataset_name, sub_dataset
                )
            if statistics is None:
                dataset = None
                try:
                    loader = get_loader_class(self.task_type()).from_datalab(
                        DatalabLoaderOption(
                            sys_info.dataset_name, sub_dataset, split=split_name
                        ),
                        output_data=None,
                    )
                    dataset = loader.load()
                except ValueError as e:
                    get_logger().warning(
                        f"{sys_info.dataset_name} could not be loaded by DataLab so"
                        " no training set dependent features will be supported by"
                        f" ExplainaBoard. Error: {e}"
                    )
                if dataset is not None:
                    statistics = self._statistics_func(dataset.samples, sys_info)
                    get_logger().info(
                        f"caching stats for {sys_info.dataset_name} {sub_dataset}"
                    )
                    write_statistics_to_cache(
                        statistics, sys_info.dataset_name, sub_dataset
                    )
        return statistics

    def _get_true_label(self, data_point: dict):
        """Get the true label from a data point.

        Returns "true_label" by default, but can be overloaded.

        Args:
            data_point: the data point under consideration

        Returns:
            the true label for the output
        """
        return data_point["true_label"]

    def _get_predicted_label(self, data_point: dict):
        """Get the predicted label from a data point.

        Returns "predicted_label" by default, but can be overloaded.

        Args:
            data_point: the data point under consideration

        Returns:
            the predicted label for the output
        """
        return data_point["predicted_label"]

    def _customize_analyses(
        self,
        sys_info: SysOutputInfo,
        custom_features: dict[str, dict[str, dict]] | None,
        metric_configs: dict[str, list[MetricConfig]] | None,
        custom_analyses: list[dict] | None,
    ) -> tuple[list[AnalysisLevel], list[Analysis]]:
        """Customize analyses for this processor.

        Args:
            custom_features: the features to customize
            metric_configs: additional metric configurations
            custom_analyses: the analyses to customize

        Returns:
            Customized analyses.
        """
        analysis_levels = self.default_analysis_levels()
        analyses = self.default_analyses()
        for level in analysis_levels:
            configs = unwrap(metric_configs)
            metric_gen = unwrap_generator(configs.get(level.name))
            for ind, metric_config in enumerate(metric_gen):
                if ind == 0:
                    level.metric_configs = [metric_config]
                else:
                    level.metric_configs.append(metric_config)
            for config in level.metric_configs:
                config.source_language = sys_info.source_language
                config.target_language = sys_info.target_language

        level_map = {x.name: x for x in analysis_levels}
        if custom_analyses is not None:
            analyses.extend([Analysis.from_dict(v) for v in custom_analyses])
        if custom_features is not None:
            ft_serializer = get_feature_type_serializer()

            for level_name, feature_content in custom_features.items():
                additional_features = {
                    k: narrow(FeatureType, ft_serializer.deserialize(v))  # type: ignore
                    if isinstance(v, dict)
                    else v
                    for k, v in feature_content.items()
                }
                level_map[level_name].features.update(additional_features)
        return analysis_levels, analyses

    @final
    def perform_analyses(
        self,
        sys_info: SysOutputInfo,
        analysis_cases: list[list[AnalysisCase]],
        metric_stats: list[list[MetricStats]],
        skip_failed_analyses: bool = False,
    ) -> list[AnalysisResult]:
        """Perform fine-grained analyses.

        Args:
            sys_info: Information about the system output
            analysis_cases: They cases to analyze
            metric_stats: The stats from which to calculate performance
            skip_failed_analyses: Whether to skip analyses when they encountered some
                errors.

        Returns:
            a dictionary of feature name -> list of performances by bucket
        """
        all_results: list[AnalysisResult] = []
        level_map = {v.name: i for i, v in enumerate(unwrap(sys_info.analysis_levels))}
        metrics = [
            [y.to_metric() for y in x.metric_configs]
            for x in unwrap(sys_info.analysis_levels)
        ]
        for my_analysis in progress(unwrap(sys_info.analyses)):
            level_id = level_map[my_analysis.level]
            try:
                all_results.append(
                    my_analysis.perform(
                        cases=analysis_cases[level_id],
                        metrics=metrics[level_id],
                        stats=metric_stats[level_id],
                        confidence_alpha=sys_info.confidence_alpha,
                    )
                )
            except Exception as ex:
                if not skip_failed_analyses:
                    raise
                get_logger().warning(f"Analysis failed, skipped. Reason: {ex}")

        return all_results

    def _gen_cases_and_stats(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        statistics: Any,
        analysis_level: AnalysisLevel,
    ) -> tuple[list[AnalysisCase], list[MetricStats]]:
        if analysis_level.name != 'example':
            raise NotImplementedError(
                f'Does not support analysis level {analysis_level.name} by default'
            )
        cases = []
        # Calculate metrics
        true_data = [self._get_true_label(x) for x in sys_output]
        pred_data = [self._get_predicted_label(x) for x in sys_output]
        metric_stats = [
            x.to_metric().calc_stats_from_data(true_data, pred_data)
            for x in analysis_level.metric_configs
        ]
        # Calculate features
        for i, output in progress(
            enumerate(sys_output), desc='calculating example-level features'
        ):
            case = AnalysisCase(sample_id=i, features={})
            for feat_name, feat_spec in analysis_level.features.items():
                if feat_spec.func is None:
                    case.features[feat_name] = output[feat_name]
                elif not feat_spec.require_training_set:
                    case.features[feat_name] = feat_spec.func(sys_info, output, case)
                elif statistics is not None:
                    case.features[feat_name] = feat_spec.func(
                        sys_info, output, case, statistics
                    )
            cases.append(case)
        return cases, metric_stats

    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        analysis_cases: list[list[AnalysisCase]],
        metric_stats: list[list[MetricStats]],
    ) -> list[list[Performance]]:
        """Get the overall performance according to metrics.

        Args:
            sys_info: Information about the system output
            analysis_cases: The cases to analyze
            metric_stats: any statistics useful to performing scoring

        Returns:
            a dictionary of metrics to overall performance numbers
        """
        overall_results = []
        for my_level, my_cases, my_stats in zip(
            unwrap(sys_info.analysis_levels), analysis_cases, metric_stats
        ):

            my_results = []
            for metric_cfg, metric_stat in zip(
                unwrap_generator(my_level.metric_configs),
                my_stats,
            ):
                metric_result = metric_cfg.to_metric().evaluate_from_stats(
                    metric_stat,
                    confidence_alpha=sys_info.confidence_alpha,
                )

                confidence_low, confidence_high = (
                    metric_result.confidence_interval
                    if metric_result.confidence_interval
                    else (None, None)
                )

                overall_performance = Performance(
                    metric_name=metric_cfg.name,
                    value=metric_result.value,
                    confidence_score_low=confidence_low,
                    confidence_score_high=confidence_high,
                )
                my_results.append(overall_performance)
            overall_results.append(my_results)
        return overall_results

    def deserialize_system_output(self, output: dict):
        """Deserialize the ystem output.

        Take a system output where the constituent data structures have been converted
        to serializable values and deserialize. By default do nothing.
        """
        return output

    @staticmethod
    def _value_by_name(bucket_perf: BucketPerformance, sort_by_metric: str) -> float:
        if sort_by_metric == 'first':
            return bucket_perf.performances[0].value
        for bp in bucket_perf.performances:
            if bp.metric_name == sort_by_metric:
                return bp.value
        raise ValueError(f'could not find metric {sort_by_metric}')

    def sort_bucket_info(
        self,
        analysis_results: list[AnalysisResult],
        sort_by: str = 'value',
        sort_by_metric: str = 'first',
        sort_ascending: bool = False,
    ) -> None:
        """Sorts the `performance_over_bucket` dictionary.

        It should be of the format
        {
            feature_name_1: {
                (bucket_1_interval_low, bucket_1_interval_up): BucketPerformance(
                    performances = [
                        Performance(metric_name = performance1),
                        ...,
                        Performance(metric_name = performancen)
                    ]
                ),
                ...
            },
            ...
        }

        Args:
            analysis_results: A list of analysis results.
            sort_by: 'key' or 'value';
                if 'key', sort by the bucket's lower boundary, alphabetically,
                low-to-high.
                if 'performance_value', sort by the `value` attribute of the
                BucketPerformance objects. Since each bucket has multiple metrics
                associated with it, see param sort_by_metric to choose which metric to
                sort on.
                if 'n_bucket_samples', sort by the number of samples in each bucket.
            sort_by_metric: 'first' or any string matching the metrics associated
                with this task.
                if 'first', sort by the value of the first BucketPerformance object,
                whichever that may be, high-to-low
                else, sort by the value of that metric.
            sort_ascending: if True, sort low-to-high; by default, sort high-to-low.
        """
        for analysis_result in analysis_results:
            if not isinstance(analysis_result, BucketAnalysisResult):
                continue
            bucket_result = analysis_result.bucket_performances

            # based on alphabetical order of the bucket lower boundary; low to high
            if sort_by == 'key':
                if bucket_result[0].bucket_interval is not None:
                    # Sort by intervals.
                    bucket_result.sort(key=lambda x: unwrap(x.bucket_interval))
                else:
                    # Sort by names.
                    bucket_result.sort(key=lambda x: unwrap(x.bucket_name))
            # sort based on the value of the first perf value, whatever that may
            # be; high to low
            elif sort_by == 'performance_value':
                bucket_result.sort(
                    key=lambda x: self._value_by_name(x, sort_by_metric),
                    reverse=not sort_ascending,
                )
            # sort by the number of samples in each bucket
            elif sort_by == 'n_bucket_samples':
                bucket_result.sort(
                    key=lambda x: x.n_samples, reverse=not sort_ascending
                )

    def get_overall_statistics(
        self, metadata: dict, sys_output: list[dict]
    ) -> OverallStatistics:
        """Get the overall statistics information of the system output.

        Args:
            metadata: The metadata of the system
            sys_output: The system output itself
        """
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = self.task_type().value

        sys_info = SysOutputInfo.from_dict(metadata)
        if sys_info.target_tokenizer is None:
            sys_info.target_tokenizer = self.get_tokenizer(sys_info.target_language)

        if sys_info.source_tokenizer is None:
            sys_info.source_tokenizer = (
                sys_info.target_tokenizer
                if sys_info.source_language == sys_info.target_language
                else self.get_tokenizer(sys_info.source_language)
            )

        # declare customized features: _features will be updated
        custom_features: dict = metadata.get('custom_features', {})
        custom_analyses: list = metadata.get('custom_analyses', [])

        metric_configs: dict[str, list[MetricConfig]] = {
            "example": metadata.get('metric_configs', [])
        }

        sys_info.analysis_levels, sys_info.analyses = self._customize_analyses(
            sys_info, custom_features, metric_configs, custom_analyses
        )

        # get scoring statistics
        external_stats = self._gen_external_stats(sys_info)

        # generate cases for each level
        analysis_cases: list[list[AnalysisCase]] = []
        metric_stats: list[list[MetricStats]] = []
        for analysis_level in unwrap(sys_info.analysis_levels):
            my_cases, my_stats = self._gen_cases_and_stats(
                sys_info, sys_output, external_stats, analysis_level
            )
            analysis_cases.append(my_cases)
            metric_stats.append(my_stats)

        # calculate overall results
        overall_results = self.get_overall_performance(
            sys_info, analysis_cases, metric_stats
        )
        sys_info.results = Result(overall=overall_results, analyses=None)
        return OverallStatistics(sys_info, analysis_cases, metric_stats)

    @final
    def process(
        self, metadata: dict, sys_output: list[dict], skip_failed_analyses: bool = False
    ) -> SysOutputInfo:
        """Run the whole process of processing the output.

        Args:
            metadata: The metadata used to specify information about processing.
            sys_output: They list of system outputs.
            skip_failed_analyses: Whether to skip failed analyses.

        Returns:
            Information about the processed system output.
        """
        overall_statistics = self.get_overall_statistics(metadata, sys_output)
        sys_info = unwrap(overall_statistics.sys_info)
        analyses = self.perform_analyses(
            sys_info,
            overall_statistics.analysis_cases,
            metric_stats=overall_statistics.metric_stats,
            skip_failed_analyses=skip_failed_analyses,
        )

        self.sort_bucket_info(
            analyses,
            sort_by=metadata.get('sort_by', 'key'),
            sort_by_metric=metadata.get('sort_by_metric', 'first'),
            sort_ascending=metadata.get('sort_ascending', False),
        )
        sys_info.results = Result(overall=sys_info.results.overall, analyses=analyses)
        return sys_info
