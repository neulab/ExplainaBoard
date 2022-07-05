from __future__ import annotations

import abc
from typing import Any, Optional

from datalabs import aggregating, Dataset, DatasetDict, load_dataset
from eaas.async_client import AsyncClient
from eaas.config import Config

from explainaboard import TaskType
from explainaboard.analysis.analyses import AnalysisLevel, AnalysisResult
from explainaboard.analysis.case import AnalysisCase
from explainaboard.analysis.performance import BucketPerformance, Performance
from explainaboard.analysis.result import Result
from explainaboard.info import OverallStatistics, SysOutputInfo
from explainaboard.metrics.metric import MetricConfig, MetricStats
from explainaboard.utils.cache_api import (
    read_statistics_from_cache,
    write_statistics_to_cache,
)
from explainaboard.utils.logging import get_logger, progress
from explainaboard.utils.tokenizer import get_default_tokenizer
from explainaboard.utils.typing_utils import unwrap, unwrap_generator


class Processor(metaclass=abc.ABCMeta):
    """Base case for task-based processor"""

    @classmethod
    @abc.abstractmethod
    def task_type(cls) -> TaskType:
        """Returns the task type of this processor."""
        ...

    @abc.abstractmethod
    def default_analyses(self) -> list[AnalysisLevel]:
        """Returns a list of analysis levels, indicating analyses that can be
        applied to different views of the higher-level example. For instance, a task may
        perform 'example'-level analysis, and 'token'-level analysis in which case this
        list would have one level for each."""
        ...

    @classmethod
    @abc.abstractmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        """Returns the default metrics of this processor."""
        ...

    @classmethod
    def full_metric_list(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        """Returns an extensive list of metrics that may be used."""
        return cls.default_metrics(
            source_language=source_language, target_language=target_language
        )

    @classmethod
    def metric_is_valid(cls, metric_config: MetricConfig) -> bool:
        """Checks if a particular metric is valid for a particular task"""
        return True

    def __init__(self) -> None:
        # Things to use only if necessary
        self._eaas_config: Optional[Config] = None
        self._eaas_client: Optional[AsyncClient] = None
        # self._statistics_func = None
        self._preprocessor = None
        # A limit on the number of samples stored for each bucket. Hard-coded for now
        self._bucket_sample_limit = 50

    def _get_statistics_resources(self, sys_info: SysOutputInfo) -> dict[str, Any]:
        """
        From a DataLab dataset split, get resources necessary to calculate statistics
        """
        return {"cls": self, "sys_info": sys_info}  #

    @aggregating
    def _statistics_func(self):
        return {}

    def _gen_external_stats(
        self, sys_info: SysOutputInfo, statistics_func: aggregating
    ):
        """Generate external statistics that are gathered from a relatively costly
        source, such as the training set.
        These are gathered once and then cached for future use.
        :param sys_info: Information about the system outputs
        :param statistics_func: The function used to get the statistics
        :return: Statistics from, usually, the training set that are used to calculate
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
                try:
                    dataset = load_dataset(sys_info.dataset_name, sub_dataset)
                except Exception:
                    dataset = None
                if dataset is None:
                    get_logger().warning(
                        f"{sys_info.dataset_name} hasn't been supported by DataLab so"
                        " no training set dependent features will be supported by"
                        " ExplainaBoard. You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md"  # noqa
                    )
                elif not (
                    isinstance(dataset, Dataset) or isinstance(dataset, DatasetDict)
                ):
                    raise ValueError(
                        'Expecting type Dataset or DatasetDict, '
                        f'but got {type(dataset)}'
                    )
                elif split_name not in dataset:
                    get_logger().warning(
                        f"{sys_info.dataset_name} has no {split_name} split in DataLab "
                        "so training set dependent features will not be calculated"
                    )
                else:
                    self._statistics_func.resources = self._get_statistics_resources(
                        sys_info
                    )
                    new_train = dataset[split_name].apply(  # type: ignore
                        self._statistics_func, mode="local"
                    )
                    statistics = new_train._stat
                    get_logger().info(
                        f"caching stats for {sys_info.dataset_name} {sub_dataset}"
                    )
                    write_statistics_to_cache(
                        statistics, sys_info.dataset_name, sub_dataset
                    )
        return statistics

    def _gen_metric_stats(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        cases: list[list[AnalysisCase]],
    ) -> list[list[MetricStats]]:
        """Generate sufficient statistics for scoring different metrics.

        :param sys_info: Information about the system outputs
        :param sys_output: The system output itself
        :param cases: Analysis cases associated with the stats
        :return: Statistics sufficient for scoring
        """
        all_stats: list[list[MetricStats]] = []
        for my_level, my_cases in zip(unwrap(sys_info.analysis_levels), cases):

            if my_level.name == 'example':
                true_data = [self._get_true_label(x) for x in sys_output]
                pred_data = [self._get_predicted_label(x) for x in sys_output]
            else:
                raise ValueError(f'unsupported analysis level {my_level.name}')

            metric_stats = []
            for metric_cfg in my_level.metric_configs:
                metric_stats.append(
                    metric_cfg.to_metric().calc_stats_from_data(true_data, pred_data)
                )

        return all_stats

    def _get_eaas_client(self):
        if not self._eaas_client:
            self._eaas_config = Config()
            self._eaas_client = AsyncClient(self._eaas_config)
        return self._eaas_client

    def _get_true_label(self, data_point: dict):
        """
        Get the true label from a data point. Returns "true_label" by default, but can
        be overloaded.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """
        return data_point["true_label"]

    def _get_predicted_label(self, data_point: dict):
        """
        Get the predicted label from a data point. Returns "predicted_label" by default,
        but can be overloaded.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["predicted_label"]

    def _customize_analyses(self, metadata: dict) -> list[AnalysisLevel]:
        """
        Customize analyses for this processor
        Args:
            metadata: the metadata information of system output

        Returns:

        """
        analyses = self.default_analyses()
        get_logger().warning('Analysis customization is not implemented yet.')
        return analyses

        # features = copy.deepcopy(self._default_features)
        # # add user-defined features into features list
        # if metadata is not None:
        #     for (
        #         feature_name,
        #         feature_config,
        #     ) in metadata.items():
        #         if feature_config["dtype"] == "string":
        #             features[feature_name] = feature.Value(
        #                 dtype="string",
        #                 description=feature_config["description"],
        #                 is_bucket=True,
        #                 is_custom=True,
        #                 bucket_info=feature.BucketInfo(
        #                     method="bucket_attribute_discrete_value",
        #                     number=feature_config["num_buckets"],
        #                     setting=1,
        #                 ),
        #             )
        #         elif feature_config["dtype"] == 'float':
        #             features[feature_name] = feature.Value(
        #                 dtype="float",
        #                 description=feature_config["description"],
        #                 is_bucket=True,
        #                 is_custom=True,
        #                 bucket_info=feature.BucketInfo(
        #                     method="bucket_attribute_specified_bucket_value",
        #                     number=feature_config["num_buckets"],
        #                     setting=(),
        #                 ),
        #             )
        #         else:
        #             raise NotImplementedError
        #
        # return features

    def perform_analyses(
        self,
        sys_info: SysOutputInfo,
        analysis_cases: list[list[AnalysisCase]],
        metric_stats: list[list[MetricStats]],
    ) -> list[list[AnalysisResult]]:
        """
        Perform fine-grained analyses
        :param sys_info: Information about the system output
        :param analysis_cases: They cases to analyze
        :param metric_stats: The stats from which to calculate performance
        :return:
            performances_over_bucket:
                a dictionary of feature name -> list of performances by bucket
        """

        all_results = []
        for my_level, my_cases, my_stats in zip(
            unwrap(sys_info.analysis_levels), analysis_cases, metric_stats
        ):
            my_results = []
            my_metrics = [x.to_metric() for x in my_level.metric_configs]
            for my_analysis in progress(
                my_level.analyses, desc=f"{my_level.name}-level analysis"
            ):
                my_results.append(
                    my_analysis.perform(
                        cases=my_cases,
                        metrics=my_metrics,
                        stats=my_stats,
                        conf_value=sys_info.conf_value,
                    )
                )
            all_results.append(my_results)

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
        """
        Get the overall performance according to metrics
        :param sys_info: Information about the system output
        :param analysis_cases: The cases to analyze
        :param metric_stats: any statistics useful to performing scoring
        :return: a dictionary of metrics to overall performance numbers
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
                    conf_value=sys_info.conf_value,
                )

                conf_low, conf_high = (
                    metric_result.conf_interval
                    if metric_result.conf_interval
                    else (None, None)
                )

                overall_performance = Performance(
                    metric_name=metric_cfg.name,
                    value=metric_result.value,
                    confidence_score_low=conf_low,
                    confidence_score_high=conf_high,
                )
                my_results.append(overall_performance)
            overall_results.append(my_results)
        return overall_results

    def deserialize_system_output(self, output: dict):
        """
        Take a system output where the constituent data structures have been converted
        to serializable values and deserialize. By default do nothing.
        """
        return output

    def sort_bucket_info(
        self,
        performance_over_bucket: dict[str, list[BucketPerformance]],
        sort_by: str = 'value',
        sort_by_metric: str = 'first',
        sort_ascending: bool = False,
    ) -> None:
        """
        Sorts the `performance_over_bucket` dictionary, which should be of the format
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

        :param sort_by: 'key' or 'value';
            if 'key', sort by the bucket's lower boundary, alphabetically, low-to-high;
            if 'performance_value', sort by the `value` attribute of the
            BucketPerformance objects. Since each bucket has multiple metrics
            associated with it, see param sort_by_metric to choose which metric to
            sort on.
            if 'n_bucket_samples', sort by the number of samples in each bucket.
        :param sort_by_metric: 'first' or any string matching the metrics associated
        with this task.
            if 'first', sort by the value of the first BucketPerformance object,
            whichever that may be, high-to-low
            else, sort by the value of that metric.
        :param sort_ascending: if True, sort low-to-high; by default, sort high-to-low.
        """

        def value_by_name(bucket_perf: BucketPerformance) -> float:
            if sort_by_metric == 'first':
                return bucket_perf.performances[0].value
            for bp in bucket_perf.performances:
                if bp.metric_name == sort_by_metric:
                    return bp.value
            raise ValueError(f'could not find metric {sort_by_metric}')

        for feature_name, feature_value in performance_over_bucket.items():

            # based on alphabetical order of the bucket lower boundary; low to high
            if sort_by == 'key':
                feature_value.sort(key=lambda x: x.bucket_interval)
            # sort based on the value of the first perf value, whatever that may
            # be; high to low
            elif sort_by == 'performance_value':
                feature_value.sort(key=value_by_name, reverse=not sort_ascending)
            # sort by the number of samples in each bucket
            elif sort_by == 'n_bucket_samples':
                feature_value.sort(
                    key=lambda x: x.n_samples, reverse=not sort_ascending
                )

    def get_overall_statistics(
        self, metadata: dict, sys_output: list[dict]
    ) -> OverallStatistics:
        """
        Get the overall statistics information, including performance, of the system
        output
        :param metadata: The metadata of the system
        :param sys_output: The system output itself
        """
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = self.task_type().value

        sys_info = SysOutputInfo.from_dict(metadata)
        get_logger().warning('Temporarily disabled metric configs')
        # if sys_info.metric_configs is None:
        #     sys_info.metric_configs = self.default_metrics(
        #         source_language=sys_info.source_language,
        #         target_language=sys_info.target_language,
        #     )
        if sys_info.target_tokenizer is None:
            sys_info.target_tokenizer = get_default_tokenizer(
                task_type=self.task_type(), lang=sys_info.target_language
            )
        if sys_info.source_tokenizer is None:
            sys_info.source_tokenizer = (
                sys_info.target_tokenizer
                if sys_info.source_language == sys_info.target_language
                else get_default_tokenizer(
                    task_type=self.task_type(), lang=sys_info.source_language
                )
            )

        # declare customized features: _features will be updated
        custom_features: dict = metadata.get('custom_features', {})
        sys_info.analysis_levels = self._customize_analyses(custom_features)

        # get scoring statistics
        external_stats = self._gen_external_stats(sys_info, self._statistics_func)

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

    def process(self, metadata: dict, sys_output: list[dict]) -> SysOutputInfo:
        overall_statistics = self.get_overall_statistics(metadata, sys_output)
        sys_info = unwrap(overall_statistics.sys_info)
        analyses = self.perform_analyses(
            sys_info,
            overall_statistics.analysis_cases,
            metric_stats=overall_statistics.metric_stats,
        )
        get_logger().warning('Sorting buckets has not been implemented')
        # self.sort_bucket_info(
        #     performance_over_bucket,
        #     sort_by=metadata.get('sort_by', 'key'),
        #     sort_by_metric=metadata.get('sort_by_metric', 'first'),
        #     sort_ascending=metadata.get('sort_ascending', False),
        # )
        sys_info.results = Result(overall=sys_info.results.overall, analyses=analyses)
        return sys_info
