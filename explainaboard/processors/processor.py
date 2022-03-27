from __future__ import annotations

import abc
from collections.abc import Mapping
import json
from typing import Any, Optional

from datalabs import aggregating, Dataset, load_dataset
from eaas.config import Config
from tqdm import tqdm

from explainaboard import feature
from explainaboard.info import (
    BucketPerformance,
    FineGrainedStatistics,
    OverallStatistics,
    Performance,
    Result,
    SysOutputInfo,
)
from explainaboard.metric import MetricStats
from explainaboard.tasks import TaskType
from explainaboard.utils.async_eaas import AsyncEaaSClient
import explainaboard.utils.bucketing
from explainaboard.utils.db_api import read_statistics_from_db, write_statistics_to_db
from explainaboard.utils.py_utils import eprint, print_dict, sort_dict
from explainaboard.utils.tokenizer import SingleSpaceTokenizer


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

    # TODO(gneubig): this could potentially be moved directly into the task definition
    @classmethod
    @abc.abstractmethod
    def default_metrics(cls) -> list[str]:
        """Returns the default metrics of this processor."""
        ...

    def __init__(self) -> None:
        # Things to use only if necessary
        self._eaas_config = None
        self._eaas_client = None
        # self._statistics_func = None
        self._tokenizer = SingleSpaceTokenizer()
        self._user_defined_feature_config = None
        self._features = self.default_features()

    def _get_statistics_resources(
        self, dataset_split: Dataset
    ) -> Optional[Mapping[str, Any]]:
        """
        From a DataLab dataset split, get resources necessary to calculate statistics
        """
        return {"cls": self}  #

    @aggregating
    def _statistics_func(self):
        return {}

    def _gen_external_stats(
        self, sys_info: SysOutputInfo, statistics_func: aggregating
    ):
        """Generate external statistics that are gathered from a relatively costly source, such as the training
        set. These are gathered once and then cached for future use.

        :param sys_info: Information about the system outputs
        :param statistics_func: The function used to get the statistics
        :return: Statistics from, usually, the training set that are used to calculate other features
        """
        statistics = None
        if sys_info.dataset_name is not None:
            split_name = "train"
            sub_dataset = (
                None
                if sys_info.sub_dataset_name == "default"
                else sys_info.sub_dataset_name
            )
            try:
                # read statistics from db
                message = None
                if sys_info.reload_stat:
                    response = read_statistics_from_db(
                        sys_info.dataset_name, sub_dataset
                    )
                    message = json.loads(response.text.replace("null", ""))["message"]
                    if message == "success":
                        return json.loads(response.content)['content']
                # calculate statistics if not reloading or not found
                if (
                    not sys_info.reload_stat
                    or message
                    == "the dataset does not include the information of _stat"
                ):
                    dataset = load_dataset(sys_info.dataset_name, sub_dataset)
                    self._statistics_func.resources = self._get_statistics_resources(
                        dataset[split_name]
                    )
                    # print(f"self._statistics_func.resources:f\t{self._statistics_func.resources}")
                    new_train = dataset[split_name].apply(
                        self._statistics_func, mode="local"
                    )
                    statistics = new_train._stat
                    eprint("saving to database")
                    response = write_statistics_to_db(
                        sys_info.dataset_name, sub_dataset, content=statistics
                    )
                    eprint(response.content)
                # dataset does not exist
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be "
                    "supported by ExplainaBoard. You can add the dataset by: "
                    "https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md"
                )
        return statistics

    def _get_metrics(self, sys_info: SysOutputInfo):
        return [getattr(explainaboard.metric, name)() for name in sys_info.metric_names]

    def _gen_metric_stats(
        self, sys_info: SysOutputInfo, sys_output: list[dict]
    ) -> list[MetricStats]:
        """Generate sufficient statistics for scoring different metrics.

        :param sys_info: Information about the system outputs
        :param sys_output: The system output itself
        :return: Statistics sufficient for scoring
        """
        metrics = self._get_metrics(sys_info)
        true_data = [self._get_true_label(x) for x in sys_output]
        pred_data = [self._get_predicted_label(x) for x in sys_output]
        metric_stats = []
        for metric in metrics:
            metric_stats.append(metric.calc_stats_from_data(true_data, pred_data))
        return metric_stats

    def _get_feature_func(self, func_name: str):
        return getattr(self, f'_get_{func_name}')

    def _get_eaas_client(self):
        if not self._eaas_client:
            self._eaas_config = Config()
            self._eaas_client = AsyncEaaSClient()
            self._eaas_client.load_config(
                self._eaas_config
            )  # The config you have created above
        return self._eaas_client

    def _get_true_label(self, data_point: dict):
        """
        Get the true label from a data point. Returns "true_label" by default, but can be overloaded.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """
        return data_point["true_label"]

    def _get_predicted_label(self, data_point: dict):
        """
        Get the predicted label from a data point. Returns "predicted_label" by default, but can be overloaded.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["predicted_label"]

    def _init_customized_features(self, metadata: dict):
        """
        declare the customized features for this processor.
        Args:
            metadata: the metadata information of system output

        Returns:

        """

        self._user_defined_feature_config = metadata.get(
            "user_defined_features_configs"
        )

        # print(f"self._user_defined_feature_config:\t{self._user_defined_feature_config}")

        # add user-defined features into features list
        if self._user_defined_feature_config is not None:
            for (
                feature_name,
                feature_config,
            ) in self._user_defined_feature_config.items():
                if feature_config["dtype"] == "string":
                    self._features[feature_name] = feature.Value(
                        dtype="string",
                        description=feature_config["description"],
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_discrete_value",
                            number=feature_config["num_buckets"],
                            setting=1,
                        ),
                    )
                elif feature_config['dtype'] == 'float':
                    self._features[feature_name] = feature.Value(
                        dtype="float",
                        description=feature_config["description"],
                        is_bucket=True,
                        bucket_info=feature.BucketInfo(
                            method="bucket_attribute_specified_bucket_value",
                            number=feature_config["num_buckets"],
                            setting=(),
                        ),
                    )
                else:
                    raise NotImplementedError

    def _complete_features(
        self, sys_info: SysOutputInfo, sys_output: list[dict], external_stats=None
    ) -> Optional[list[str]]:
        """
        This function takes in meta-data about system outputs, system outputs, and a few other optional pieces of
        information, then calculates feature functions and modifies `sys_output` to add these feature values

        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param external_stats: Extenral statistics that are used to calculate training set specific features
        :return: The features that are active (e.g. skipping training set features when no training set available)
        """
        # Get names of bucketing features
        bucket_feature_funcs = {}
        for bucket_feature in sys_info.features.get_bucket_features():

            # handles user-defined features
            if (
                self._user_defined_feature_config is not None
                and bucket_feature in self._user_defined_feature_config.keys()
                and (
                    external_stats is not None
                    or not sys_info.features[bucket_feature].require_training_set
                )
            ):
                bucket_feature_funcs[bucket_feature] = (
                    None,  # no need to call a function for user-defined features; they are already in the data point itself
                    sys_info.features[bucket_feature].require_training_set,
                )

            # handles all other features
            elif bucket_feature in sys_info.features.keys() and (
                external_stats is not None
                or not sys_info.features[bucket_feature].require_training_set
            ):
                bucket_feature_funcs[bucket_feature] = (
                    self._get_feature_func(bucket_feature),
                    sys_info.features[bucket_feature].require_training_set,
                )
        for _id, dict_sysout in tqdm(enumerate(sys_output), desc="featurizing"):
            # Get values of bucketing features
            for (
                bucket_key,
                (
                    bucket_func,
                    training_dependent,
                ),
            ) in bucket_feature_funcs.items():

                # handles user-defined features
                if (
                    self._user_defined_feature_config is not None
                    and bucket_key in self._user_defined_feature_config.keys()
                ):
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
                        bucket_func(dict_sysout, external_stats)
                        if training_dependent
                        else bucket_func(dict_sysout)
                    )
        return list(bucket_feature_funcs.keys())

    def _bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        active_features: list[str],
        metric_stats=None,
    ) -> tuple[dict, dict]:
        """
        Separate samples into buckets and calculate performance over them
        :param sys_info: Information about the system output
        :param sys_output: The system output itself, already annotated with features
        :return:
            samples_over_bucket: a dictionary of feature name -> list of buckets and samples
            performances_over_bucket: a dictionary of feature name -> list of performances by bucket
        """

        # Bucketing
        samples_over_bucket = {}
        performances_over_bucket = {}
        for feature_name in tqdm(active_features, desc="bucketing"):
            # print(f"Feature Name: {feature_name}\n"
            #       f"Bucket Hyper:\n function_name: {sys_info.features[feature_name].bucket_info.method} \n"
            #       f"bucket_number: {sys_info.features[feature_name].bucket_info.number}\n"
            #       f"bucket_setting: {sys_info.features[feature_name].bucket_info.setting}\n")

            # Preparation for bucketing
            bucket_func = getattr(
                explainaboard.utils.bucketing,
                sys_info.features[feature_name].bucket_info.method,
            )
            # TODO(gneubig): make dict_obj more elegant so it doesn't have to copy memory
            samples_over_bucket[feature_name] = bucket_func(
                dict_obj={
                    x: sys_output[x][feature_name] for x in range(len(sys_output))
                },
                bucket_number=sys_info.features[feature_name].bucket_info.number,
                bucket_setting=sys_info.features[feature_name].bucket_info.setting,
            )

            # evaluating bucket: get bucket performance
            performances_over_bucket[feature_name] = self.get_bucket_performance(
                sys_info,
                sys_output,
                samples_over_bucket[feature_name],
                metric_stats=metric_stats,
            )

        return samples_over_bucket, performances_over_bucket

    def get_bucket_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        samples_over_bucket: dict[str, list[int]],
        metric_stats: list[MetricStats] = None,
    ) -> dict[str, list[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample IDs for that bucket
        :param metric_stats: any statistics useful to performing scoring
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        # Get the functions to calculate metrics
        metric_funcs = self._get_metrics(sys_info)

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in samples_over_bucket.items():

            bucket_cases = []

            for sample_id in sample_ids:

                data_point = sys_output[sample_id]
                true_label = self._get_true_label(data_point)
                predicted_label = self._get_predicted_label(data_point)
                s_id = data_point["id"]

                # get a bucket of cases (e.g., errors)
                if sys_info.is_print_case:
                    if true_label != predicted_label:
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name, metric_func, metric_stat in zip(
                sys_info.metric_names, metric_funcs, metric_stats
            ):
                bucket_stats = metric_stat.filter(sample_ids)
                metric_result = metric_func.evaluate_from_stats(
                    bucket_stats,
                    conf_value=0.05 if sys_info.is_print_confidence_interval else None,
                )

                conf_low, conf_high = (
                    metric_result.conf_interval
                    if metric_result.conf_interval
                    else (None, None)
                )

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=metric_result.value,
                    confidence_score_low=conf_low,
                    confidence_score_high=conf_high,
                    n_samples=len(bucket_stats),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)

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
        for metric_name, metric_func, metric_stat in zip(
            sys_info.metric_names, metric_funcs, metric_stats
        ):
            metric_result = metric_func.evaluate_from_stats(
                metric_stat,
                conf_value=0.05 if sys_info.is_print_confidence_interval else None,
            )

            conf_low, conf_high = (
                metric_result.conf_interval
                if metric_result.conf_interval
                else (None, None)
            )

            overall_performance = Performance(
                metric_name=metric_name,
                value=metric_result.value,
                confidence_score_low=conf_low,
                confidence_score_high=conf_high,
            )
            overall_results[metric_name] = overall_performance
        return overall_results

    def _print_bucket_info(
        self, performances_over_bucket: dict[str, dict[str, list[BucketPerformance]]]
    ):
        """
        Print out performance bucket by bucket
        :param performances_over_bucket: dictionary of features -> buckets -> performance for different metrics
        """
        for feature_name, feature_value in performances_over_bucket.items():
            print_dict(feature_value, feature_name)

    def get_overall_statistics(
        self, metadata: dict, sys_output: list[dict]
    ) -> OverallStatistics:
        """
        Get the overall statistics information, including performance, of the system output
        :param metadata: The metadata of the system
        :param sys_output: The system output itself
        """
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = self.task_type().value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = self.default_metrics()
        sys_info = SysOutputInfo.from_dict(metadata)

        # declare customized features: _features will be updated
        self._init_customized_features(metadata)

        sys_info.features = self._features

        # get scoring statistics
        metric_stats = self._gen_metric_stats(sys_info, sys_output)
        external_stats = self._gen_external_stats(sys_info, self._statistics_func)
        active_features = self._complete_features(
            sys_info, sys_output, external_stats=external_stats
        )
        overall_results = self.get_overall_performance(
            sys_info, sys_output, metric_stats=metric_stats
        )
        return OverallStatistics(
            sys_info, metric_stats, active_features, overall_results
        )

    def get_fine_grained_statistics(
        self,
        sys_info: SysOutputInfo,
        sys_output: list[dict],
        active_features: list[str],
        metric_stats=None,
    ) -> tuple[dict, dict]:
        samples_over_bucket, performance_over_bucket = self._bucketing_samples(
            sys_info, sys_output, active_features, metric_stats
        )
        """
        A wrapper function to expose _bucketing_samples for the web interface
        """
        return FineGrainedStatistics(samples_over_bucket, performance_over_bucket)

    def process(self, metadata: dict, sys_output: list[dict]):
        # TODO(Pengfei): Rethink if this is a good way to manipulate `system_output`
        overall_statistics = self.get_overall_statistics(metadata, sys_output)
        sys_info = overall_statistics.sys_info
        metric_stats = overall_statistics.metric_stats
        active_features = overall_statistics.active_features
        overall_results = overall_statistics.overall_results
        samples_over_bucket, performance_over_bucket = self._bucketing_samples(
            sys_info, sys_output, active_features, metric_stats=metric_stats
        )
        self._print_bucket_info(performance_over_bucket)
        sys_info.results = Result(
            overall=overall_results, fine_grained=performance_over_bucket
        )
        return sys_info
