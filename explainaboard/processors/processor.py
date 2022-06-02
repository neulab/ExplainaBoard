from __future__ import annotations

import abc
from collections.abc import Callable
import copy
from typing import Any, Optional, TypeVar

from datalabs import aggregating, Dataset, DatasetDict, load_dataset
from eaas.async_client import AsyncClient
from eaas.config import Config
import numpy as np

from explainaboard import feature, TaskType
from explainaboard.feature import Features
from explainaboard.info import (
    BucketCase,
    BucketCaseCollection,
    BucketPerformance,
    OverallStatistics,
    Performance,
    print_bucket_perfs,
    Result,
    SysOutputInfo,
)
from explainaboard.metric import Metric, MetricConfig, MetricStats
import explainaboard.utils.bucketing
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

    @classmethod
    @abc.abstractmethod
    def default_features(cls) -> feature.Features:
        """Returns default features for this processor."""
        ...

    @classmethod
    @abc.abstractmethod
    def default_metrics(
        cls, source_language=None, target_language=None
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
        self._default_features = self.default_features()
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

    def _get_metrics(self, sys_info: SysOutputInfo) -> list[Metric]:
        return [config.to_metric() for config in unwrap(sys_info.metric_configs)]

    def _gen_metric_stats(
        self, sys_info: SysOutputInfo, sys_output: list[dict]
    ) -> list[MetricStats]:
        """Generate sufficient statistics for scoring different metrics.

        :param sys_info: Information about the system outputs
        :param sys_output: The system output itself
        :return: Statistics sufficient for scoring
        """
        metrics = unwrap(self._get_metrics(sys_info))
        true_data = [self._get_true_label(x) for x in sys_output]
        pred_data = [self._get_predicted_label(x) for x in sys_output]
        metric_stats = []

        for metric in metrics:
            metric_stats.append(metric.calc_stats_from_data(true_data, pred_data))
        return metric_stats

    def _get_feature_func(self, feature_name: str, is_custom: bool):
        if is_custom:

            def my_func(info, sysout, stats=None):
                return sysout[feature_name]

            return my_func
        else:
            return getattr(self, f'_get_{feature_name}')

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

    def _customize_features(self, metadata: dict) -> Optional[Features]:
        """
        declare the customized features for this processor.
        Args:
            metadata: the metadata information of system output

        Returns:

        """
        features = copy.deepcopy(self._default_features)

        # add user-defined features into features list
        if metadata is not None:
            for (
                feature_name,
                feature_config,
            ) in metadata.items():
                if feature_config["dtype"] == "string":
                    features[feature_name] = feature.Value(
                        dtype="string",
                        description=feature_config["description"],
                        is_bucket=True,
                        is_custom=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_discrete_value",
                            number=feature_config["num_buckets"],
                            setting=1,
                        ),
                    )
                elif feature_config["dtype"] == 'float':
                    features[feature_name] = feature.Value(
                        dtype="float",
                        description=feature_config["description"],
                        is_bucket=True,
                        is_custom=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=feature_config["num_buckets"],
                            setting=(),
                        ),
                    )
                else:
                    raise NotImplementedError

        return features

    def _complete_features(
        self, sys_info: SysOutputInfo, sys_output: list[dict], external_stats=None
    ) -> list[str]:
        """
        This function takes in meta-data about system outputs, system outputs, and a few
        other optional pieces of information, then calculates feature functions and
        modifies `sys_output` to add these feature values

        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param external_stats: External statistics that are used to calculate training
            set specific features
        :return: The features that are active (e.g. skipping training set features when
            no training set available)
        """
        bucket_feature_funcs: dict[str, tuple[Callable, bool]] = {}
        sys_features = unwrap(sys_info.features)

        for bucket_feature in sys_features.get_bucket_features():

            feature_info = sys_features[bucket_feature]

            # Skip training set features if no stats
            if external_stats is None and feature_info.require_training_set:
                continue

            feature_func = self._get_feature_func(
                bucket_feature, feature_info.is_custom
            )

            bucket_feature_funcs[bucket_feature] = (
                feature_func,
                feature_info.require_training_set,
            )

        for _id, dict_sysout in progress(enumerate(sys_output), desc="featurizing"):
            # Get values of bucketing features
            for (
                bucket_key,
                (
                    bucket_func,
                    training_dependent,
                ),
            ) in bucket_feature_funcs.items():

                feature_info = sys_features[bucket_key]

                # handles user-defined features
                if feature_info.is_custom:
                    # TODO(Pengfei): this should be generalized
                    feature_value = (
                        "_".join(dict_sysout[bucket_key])
                        if isinstance(dict_sysout[bucket_key], list)
                        else dict_sysout[bucket_key]
                    )
                    dict_sysout[bucket_key] = feature_value

                # handles all other features
                else:
                    dict_sysout[bucket_key] = (
                        bucket_func(sys_info, dict_sysout, external_stats)
                        if training_dependent
                        else bucket_func(sys_info, dict_sysout)
                    )

        return list(bucket_feature_funcs.keys())

    def bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        active_features: list[str],
        metric_stats: list[MetricStats],
    ) -> dict[str, list[BucketPerformance]]:
        """
        Separate samples into buckets and calculate performance over them
        :param sys_info: Information about the system output
        :param sys_output: The system output itself, already annotated with features
        :param active_features: The features to perform bucketing over
        :param metric_stats: The stats from which to calculate performance
        :return:
            performances_over_bucket:
                a dictionary of feature name -> list of performances by bucket
        """
        sys_features = unwrap(sys_info.features)

        # Bucketing
        performances_over_bucket: dict[str, list[BucketPerformance]] = {}
        for feature_name in progress(active_features, desc="sample-level bucketing"):
            # Preparation for bucketing
            bucket_func: Callable[..., list[BucketCaseCollection]] = getattr(
                explainaboard.utils.bucketing,
                sys_features[feature_name].bucket_info.method,
            )
            samples_over_bucket = bucket_func(
                sample_features=[
                    (BucketCase(x), sys_output[x][feature_name])
                    for x in range(len(sys_output))
                ],
                bucket_number=sys_features[feature_name].bucket_info.number,
                bucket_setting=sys_features[feature_name].bucket_info.setting,
            )

            # evaluating bucket: get bucket performance
            performances_over_bucket[feature_name] = self.get_bucket_performance(
                sys_info,
                sys_output,
                samples_over_bucket,
                metric_stats=metric_stats,
            )
        return performances_over_bucket

    BucketCaseType = TypeVar('BucketCaseType')

    def _subsample_bucket_cases(
        self, bucket_cases: list[BucketCaseType]
    ) -> list[BucketCaseType]:
        if len(bucket_cases) > self._bucket_sample_limit:
            ids = np.array(range(len(bucket_cases)))
            bucket_sample_ids = np.random.choice(
                ids, self._bucket_sample_limit, replace=False
            )
            return [bucket_cases[i] for i in bucket_sample_ids]
        else:
            return bucket_cases

    def get_bucket_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        samples_over_bucket: list[BucketCaseCollection],
        metric_stats: list[MetricStats],
    ) -> list[BucketPerformance]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature
        (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample
            IDs for that bucket
        :param metric_stats: any statistics useful to performing scoring
        :return: bucket_name_to_performance: a dictionary that maps bucket names to
            bucket performance
        """

        # Get the functions to calculate metrics
        metric_funcs = self._get_metrics(sys_info)

        bucket_performances: list[BucketPerformance] = []
        for bucket_collection in samples_over_bucket:
            bucket_cases = bucket_collection.samples
            sample_ids = [bucket_case.sample_id for bucket_case in bucket_cases]
            # Subsample examples to save
            bucket_samples = self._subsample_bucket_cases(bucket_cases)

            bucket_performance = BucketPerformance(
                bucket_interval=bucket_collection.interval,
                n_samples=len(bucket_cases),
                bucket_samples=bucket_samples,
            )

            for metric_cfg, metric_func, metric_stat in zip(
                unwrap_generator(sys_info.metric_configs),
                unwrap_generator(metric_funcs),
                unwrap_generator(metric_stats),
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

        return bucket_performances

    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        metric_stats: list[MetricStats],
    ) -> dict[str, Performance]:
        """
        Get the overall performance according to metrics
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param metric_stats: any statistics useful to performing scoring
        :return: a dictionary of metrics to overall performance numbers
        """
        predicted_labels, true_labels = [], []

        for _id, feature_table in enumerate(sys_output):
            predicted_labels.append(self._get_predicted_label(feature_table))
            true_labels.append(self._get_true_label(feature_table))

        metric_funcs = self._get_metrics(sys_info)

        overall_results = {}
        for metric_cfg, metric_func, metric_stat in zip(
            unwrap_generator(sys_info.metric_configs),
            unwrap_generator(metric_funcs),
            metric_stats,
        ):
            metric_result = metric_func.evaluate_from_stats(
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
            overall_results[metric_cfg.name] = overall_performance
        return overall_results

    def deserialize_system_output(self, output: dict):
        """
        Take a system output where the constituent data structures have been converted
        to serializable values and deserialize. By default do nothing.
        """
        return output

    def print_bucket_info(
        self, performances_over_bucket: dict[str, list[BucketPerformance]]
    ):
        """
        Print out performance bucket by bucket
        :param performances_over_bucket:
            dictionary of features -> buckets -> performance for different metrics
        """
        for feature_name, feature_value in performances_over_bucket.items():
            print_bucket_perfs(feature_value, feature_name)

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
        if sys_info.metric_configs is None:
            sys_info.metric_configs = self.default_metrics(
                source_language=sys_info.source_language,
                target_language=sys_info.target_language,
            )
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
        sys_info.features = self._customize_features(custom_features)

        # get scoring statistics
        metric_stats = unwrap(self._gen_metric_stats(sys_info, sys_output))
        external_stats = self._gen_external_stats(sys_info, self._statistics_func)
        active_features = self._complete_features(
            sys_info, sys_output, external_stats=external_stats
        )
        overall_results = self.get_overall_performance(
            sys_info, sys_output, metric_stats=metric_stats
        )
        sys_info.results = Result(
            overall=overall_results, calibration=None, fine_grained=None
        )
        return OverallStatistics(sys_info, metric_stats, active_features)

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

    def process(self, metadata: dict, sys_output: list[dict]) -> SysOutputInfo:
        # TODO(Pengfei): Rethink if this is a good way to manipulate `system_output`
        overall_statistics = self.get_overall_statistics(metadata, sys_output)
        sys_info = unwrap(overall_statistics.sys_info)
        metric_stats = overall_statistics.metric_stats
        active_features = unwrap(overall_statistics.active_features)
        overall_results = sys_info.results.overall
        performance_over_bucket = self.bucketing_samples(
            sys_info, sys_output, active_features, metric_stats=metric_stats
        )
        self.sort_bucket_info(
            performance_over_bucket,
            sort_by=metadata.get('sort_by', 'key'),
            sort_by_metric=metadata.get('sort_by_metric', 'first'),
            sort_ascending=metadata.get('sort_ascending', False),
        )
        sys_info.results = Result(
            overall=overall_results, fine_grained=performance_over_bucket
        )
        return sys_info
