"""A parent class to represent processors."""

from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Any, cast, final, Optional

from eaas.async_client import AsyncClient
from eaas.config import Config

from explainaboard import TaskType
from explainaboard.analysis.analyses import (
    Analysis,
    AnalysisLevel,
    AnalysisResult,
    BucketAnalysis,
    BucketAnalysisDetails,
    CalibrationAnalysis,
)
from explainaboard.analysis.case import AnalysisCase
from explainaboard.analysis.feature import DataType, FeatureType, Value
from explainaboard.analysis.result import Result
from explainaboard.info import OverallStatistics, SysOutputInfo
from explainaboard.loaders import DatalabLoaderOption, get_loader_class
from explainaboard.metrics.metric import MetricConfig, MetricResult, MetricStats, Score
from explainaboard.utils.cache_api import (
    read_statistics_from_cache,
    write_statistics_to_cache,
)
from explainaboard.utils.logging import get_logger, progress
from explainaboard.utils.tokenizer import get_default_tokenizer, Tokenizer
from explainaboard.utils.typing_utils import narrow, unwrap


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
                if (
                    isinstance(v, Value)
                    and v.dtype == DataType.FLOAT
                    and not v.optional
                ):
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
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """Returns the default metrics of this processor.

        Args:
            level: Name of the analysis level.
            source_language: Source language code.
            target_language: Target language code.

        Returns:
            Mapping of metric name -> MetricConfig.
        """
        ...

    @classmethod
    def full_metric_list(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """Returns an extensive list of metrics that may be used.

        Args:
            level: Name of the analysis level.
            source_language: Source language code.
            target_language: Target language code.

        Returns:
            Mapping of metric name -> MetricConfig.
        """
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
    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo) -> Any:
        ...

    def _gen_external_stats(self, sys_info: SysOutputInfo, use_cache: bool) -> Any:
        """Generate external statistics.

        These are gathered from a relatively costly source, such as the training set,
        then cached for future use.

        Args:
            sys_info: Information about the system outputs
            use_cache: whether to reload the statistics from cache or not.

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
            if use_cache:
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
                except FileNotFoundError as e:
                    get_logger().warning(
                        f"{sys_info.dataset_name} could not be loaded by DataLab so"
                        " no training set dependent features will be supported by"
                        f" ExplainaBoard. Error: {e}"
                    )
                except ValueError as e:
                    get_logger().warning(
                        f"Data split `{split_name}` couldn't been found. Error: {e}"
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
        custom_features: dict[str, dict[str, FeatureType]],
        metric_configs: dict[str, dict[str, MetricConfig]],
        custom_analyses: list[Analysis],
    ) -> tuple[list[AnalysisLevel], list[Analysis]]:
        """Customize analyses for this processor.

        Args:
            custom_features: the features to customize
            metric_configs: MetricConfgs to replace.
                If `metric_configs[analysis_level_name]` has a dict, it is used instead
                of the default MetricConfigs associated to `analysis_level_name`.
            custom_analyses: the analyses to customize

        Returns:
            Customized analyses.
        """
        analysis_levels: list[AnalysisLevel] = []

        # Replaces MetricConfigs for each AnalysisLevel.
        for level in self.default_analysis_levels():
            metric_configs_orig = metric_configs.get(level.name, level.metric_configs)
            metric_configs_replaced = {
                name: config.replace_languages(
                    source_language=sys_info.source_language,
                    target_language=sys_info.target_language,
                )
                for name, config in metric_configs_orig.items()
            }
            analysis_levels.append(
                level.replace_metric_configs(metric_configs_replaced)
            )

        level_map = {x.name: x for x in analysis_levels}

        analyses = self.default_analyses()
        analyses.extend(custom_analyses)

        for level_name, feature_content in custom_features.items():
            level_map[level_name].features.update(feature_content)

        return analysis_levels, analyses

    @final
    def perform_analyses(
        self,
        sys_info: SysOutputInfo,
        analysis_cases: list[list[AnalysisCase]],
        metric_stats: list[dict[str, MetricStats]],
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
        level_map = {v.name: i for i, v in enumerate(sys_info.analysis_levels)}
        metrics = [
            {name: config.to_metric() for name, config in level.metric_configs.items()}
            for level in sys_info.analysis_levels
        ]
        for my_analysis in progress(sys_info.analyses):
            level_id = level_map[my_analysis.level]
            try:
                if (
                    isinstance(my_analysis, CalibrationAnalysis)
                    and my_analysis.feature not in analysis_cases[level_id][0].features
                ):
                    continue

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
    ) -> tuple[list[AnalysisCase], dict[str, MetricStats]]:
        """Generates analysis cases and stats.

        Args:
            sys_info: Information about the system output.
            sys_output: TBD
            statistics: TBD
            analysis_level: Analysis level corresponding to the returned information.

        Returns:
            Tuple of following values:
                - List of analysis levels.
                - Mapping from metric name to stats.
        """
        if analysis_level.name != "example":
            raise NotImplementedError(
                f"Does not support analysis level {analysis_level.name} by default"
            )

        # Calculate metrics
        true_data = [self._get_true_label(x) for x in sys_output]
        pred_data = [self._get_predicted_label(x) for x in sys_output]
        metric_stats = {
            name: config.to_metric().calc_stats_from_data(true_data, pred_data)
            for name, config in analysis_level.metric_configs.items()
        }

        # Calculate features
        cases: list[AnalysisCase] = []
        for i, output in progress(
            enumerate(sys_output), desc="calculating example-level features"
        ):
            case = AnalysisCase(sample_id=i, features={})
            for feat_name, feat_spec in analysis_level.features.items():
                if feat_name not in output and feat_spec.optional:
                    continue
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
        metric_stats: list[dict[str, MetricStats]],
    ) -> dict[str, dict[str, MetricResult]]:
        """Get the overall performance according to metrics.

        Args:
            sys_info: Information about the system output
            analysis_cases: The cases to analyze
            metric_stats: any statistics useful to performing scoring

        Returns:
            a dictionary of metrics to overall performance numbers
        """
        overall_results: dict[str, dict[str, MetricResult]] = {}

        for my_level, my_stats in zip(sys_info.analysis_levels, metric_stats):
            my_results: dict[str, MetricResult] = {}

            for metric_name, metric_cfg in my_level.metric_configs.items():
                metric_stat = my_stats[metric_name]
                my_results[metric_name] = metric_cfg.to_metric().evaluate_from_stats(
                    metric_stat,
                    confidence_alpha=sys_info.confidence_alpha,
                )

            overall_results[my_level.name] = my_results

        return overall_results

    def deserialize_system_output(self, output: dict):
        """Deserialize the ystem output.

        Take a system output where the constituent data structures have been converted
        to serializable values and deserialize. By default do nothing.
        """
        return output

    def sort_bucket_info(
        self,
        analysis_results: list[AnalysisResult],
        sort_by: str,
        sort_by_metric: str | None = None,
        sort_ascending: bool = False,
    ) -> None:
        """Sorts the `performance_over_bucket` dictionary.

        Args:
            analysis_results: A list of analysis results.
            sort_by: 'key', 'performance_value', or 'n_bucket_samples';
                if 'key', sort by the bucket's lower boundary, alphabetically,
                low-to-high.
                if 'performance_value', sort by the `value` attribute of the
                BucketPerformance objects. Since each bucket has multiple metrics
                associated with it, see param sort_by_metric to choose which metric to
                sort on.
                if 'n_bucket_samples', sort by the number of samples in each bucket.
            sort_by_metric: Key of the metric name, or None. This argument must be set
                when and only when `sort_by` is `performance_value`.
                else, sort by the value of that metric.
            sort_ascending: if True, sort low-to-high; by default, sort high-to-low.
        """
        for analysis_result in analysis_results:
            if not isinstance(analysis_result.details, BucketAnalysisDetails):
                continue

            bucket_result = analysis_result.details.bucket_performances

            # based on alphabetical order of the bucket lower boundary; low to high
            if sort_by == "key":
                if bucket_result[0].bucket_interval is not None:
                    # Sort by intervals.
                    bucket_result.sort(key=lambda x: unwrap(x.bucket_interval))
                else:
                    # Sort by names.
                    bucket_result.sort(key=lambda x: unwrap(x.bucket_name))
            # sort based on the value of the first perf value, whatever that may
            # be; high to low
            elif sort_by == "performance_value":
                if sort_by_metric is None:
                    raise ValueError("sort_by_metric must be set.")
                bucket_result.sort(
                    key=lambda x: x.results[cast(str, sort_by_metric)]
                    .get_value(Score, "score")
                    .value,
                    reverse=not sort_ascending,
                )
            # sort by the number of samples in each bucket
            elif sort_by == "n_bucket_samples":
                bucket_result.sort(
                    key=lambda x: x.n_samples, reverse=not sort_ascending
                )
            else:
                raise ValueError(f"Invalid sort_by: {sort_by}")

    def get_overall_statistics(
        self,
        metadata: dict,
        sys_output: list[dict],
        use_cache: bool = True,
    ) -> OverallStatistics:
        """Get the overall statistics information of the system output.

        Args:
            metadata: The metadata of the system
            sys_output: The system output itself
            use_cache: whether to reload the statistics from cache or not.
        """
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = self.task_type().value

        sys_info = SysOutputInfo.from_any_dict(metadata)

        if sys_info.target_tokenizer is None:
            sys_info.target_tokenizer = self.get_tokenizer(sys_info.target_language)

        if sys_info.source_tokenizer is None:
            sys_info.source_tokenizer = (
                sys_info.target_tokenizer
                if sys_info.source_language == sys_info.target_language
                else self.get_tokenizer(sys_info.source_language)
            )

        # declare customized features: _features will be updated
        custom_features: dict[str, dict[str, FeatureType]] = metadata.get(
            "custom_features", {}
        )
        custom_analyses: list[Analysis] = metadata.get("custom_analyses", [])

        metric_configs = metadata.get("metric_configs")
        if metric_configs is not None:
            metric_configs_dict = {
                "example": {
                    narrow(str, k): narrow(MetricConfig, v)  # type: ignore
                    for k, v in metric_configs.items()
                }
            }
        else:
            metric_configs_dict = {}

        sys_info.analysis_levels, sys_info.analyses = self._customize_analyses(
            sys_info, custom_features, metric_configs_dict, custom_analyses
        )

        # get scoring statistics
        external_stats = self._gen_external_stats(sys_info, use_cache)

        # generate cases for each level
        analysis_cases: list[list[AnalysisCase]] = []
        metric_stats: list[dict[str, MetricStats]] = []
        for analysis_level in sys_info.analysis_levels:
            my_cases, my_stats = self._gen_cases_and_stats(
                sys_info, sys_output, external_stats, analysis_level
            )
            analysis_cases.append(my_cases)
            metric_stats.append(my_stats)

        # calculate overall results
        overall_results = self.get_overall_performance(sys_info, metric_stats)
        sys_info.results = Result(overall=overall_results, analyses=[])
        return OverallStatistics(sys_info, analysis_cases, metric_stats)

    @final
    def process(
        self,
        metadata: dict,
        sys_output: list[dict],
        skip_failed_analyses: bool = False,
        use_cache: bool = True,
    ) -> SysOutputInfo:
        """Run the whole process of processing the output.

        Args:
            metadata: The metadata used to specify information about processing.
            sys_output: They list of system outputs.
            skip_failed_analyses: Whether to skip failed analyses.
            use_cache: whether to reload the statistics or not.

        Returns:
            Information about the processed system output.
        """
        overall_statistics = self.get_overall_statistics(
            metadata,
            sys_output,
            use_cache,
        )
        sys_info = unwrap(overall_statistics.sys_info)
        analyses = self.perform_analyses(
            sys_info,
            overall_statistics.analysis_cases,
            metric_stats=overall_statistics.metric_stats,
            skip_failed_analyses=skip_failed_analyses,
        )

        self.sort_bucket_info(
            analyses,
            sort_by=metadata.get("sort_by", "key"),
            sort_by_metric=metadata.get("sort_by_metric"),
            sort_ascending=metadata.get("sort_ascending", False),
        )
        sys_info.results = Result(overall=sys_info.results.overall, analyses=analyses)
        return sys_info
