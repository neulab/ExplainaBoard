import json
from typing import List, Tuple, Dict, Optional, Mapping, Any

from datalabs import load_dataset, aggregating, Dataset

from explainaboard.utils.async_eaas import AsyncEaaSClient
from eaas.config import Config
from tqdm import tqdm

import explainaboard.metric
import explainaboard.utils.bucketing
from explainaboard import feature
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance, Result
from explainaboard.tasks import TaskType
from explainaboard.utils.db_api import read_statistics_from_db, write_statistics_to_db
from explainaboard.utils.py_utils import (
    eprint,
    print_dict,
    sort_dict,
)
from explainaboard.utils.tokenizer import SingleSpaceTokenizer


class Processor:
    """Base case for task-based processor"""

    _features: feature.Features
    _task_type: TaskType
    # TODO(gneubig): this could potentially be moved directly into the task definition
    _default_metrics: List[str]

    def __init__(self) -> None:
        # Things to use only if necessary
        self._eaas_config = None
        self._eaas_client = None
        self._statistics_func = None
        self._tokenizer = SingleSpaceTokenizer()
        self._user_defined_feature_config = None

    def _get_urces(self, dataset_split: Dataset) -> Optional[Mapping[str, Any]]:
        """
        From a DataLab dataset split, get resources necessary to calculate statistics
        """
        return None

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
                    statistics_func.resources = self._get_urces(dataset)
                    new_train = dataset[split_name].apply(statistics_func, mode="local")
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

    def _gen_scoring_stats(
        self, sys_info: SysOutputInfo, sys_output: List[dict]
    ) -> Any:
        """Generate sufficient statistics for scoring.

        :param sys_info: Information about the system outputs
        :param sys_output: The system output itself
        :return: Statistics sufficient for scoring
        """
        return None

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

    def _complete_features(
        self, sys_info: SysOutputInfo, sys_output: List[dict], external_stats=None
    ) -> List[str]:
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
                    feature_value = dict_sysout[bucket_key]
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
        sys_output: List[dict],
        active_features: List[str],
        scoring_stats=None,
    ) -> Tuple[dict, dict]:
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
                scoring_stats=scoring_stats,
            )

        return samples_over_bucket, performances_over_bucket

    def get_bucket_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        samples_over_bucket: Dict[str, List[int]],
        scoring_stats: Any = None,
    ) -> Dict[str, List[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample IDs for that bucket
        :param scoring_stats: any statistics useful to performing scoring
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in samples_over_bucket.items():

            bucket_true_labels = []
            bucket_predicted_labels = []
            bucket_cases = []

            for sample_id in sample_ids:

                data_point = sys_output[sample_id]
                true_label = self._get_true_label(data_point)
                predicted_label = self._get_predicted_label(data_point)
                s_id = data_point["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if sys_info.is_print_case:
                    if true_label != predicted_label:
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in sys_info.metric_names:
                metric_func = getattr(explainaboard.metric, metric_name)
                one_metric = metric_func(
                    true_labels=bucket_true_labels,
                    predicted_labels=bucket_predicted_labels,
                    is_print_confidence_interval=sys_info.is_print_confidence_interval,
                )
                metric_result = one_metric.evaluate()

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=metric_result["value"],
                    confidence_score_low=metric_result["confidence_score_low"],
                    confidence_score_high=metric_result["confidence_score_high"],
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)

    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        scoring_stats: Any = None,
    ) -> Dict[str, Performance]:
        """
        Get the overall performance according to metrics
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param scoring_stats: any statistics useful to performing scoring
        :return: a dictionary of metrics to overall performance numbers
        """
        predicted_labels, true_labels = [], []

        for _id, feature_table in enumerate(sys_output):

            predicted_labels.append(self._get_predicted_label(feature_table))
            true_labels.append(self._get_true_label(feature_table))

        overall_results = {}
        for metric_name in sys_info.metric_names:
            metric_func = getattr(explainaboard.metric, metric_name)
            one_metric = metric_func(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                is_print_confidence_interval=sys_info.is_print_confidence_interval,
            )
            metric_result = one_metric.evaluate()

            overall_performance = Performance(
                metric_name=metric_name,
                value=metric_result["value"],
                confidence_score_low=metric_result["confidence_score_low"],
                confidence_score_high=metric_result["confidence_score_high"],
            )
            overall_results[metric_name] = overall_performance
        return overall_results

    def _print_bucket_info(
        self, performances_over_bucket: Dict[str, Dict[str, List[BucketPerformance]]]
    ):
        """
        Print out performance bucket by bucket
        :param performances_over_bucket: dictionary of features -> buckets -> performance for different metrics
        """
        for feature_name, feature_value in performances_over_bucket.items():
            print_dict(feature_value, feature_name)

    def process(self, metadata: dict, sys_output: List[dict]) -> SysOutputInfo:
        if metadata is None:
            metadata = {}
        if "task_name" not in metadata.keys():
            metadata["task_name"] = self._task_type.value
        if "metric_names" not in metadata.keys():
            metadata["metric_names"] = self._default_metrics
        sys_info = SysOutputInfo.from_dict(metadata)
        sys_info.features = self._features
        scoring_stats = self._gen_scoring_stats(sys_info, sys_output)
        external_stats = self._gen_external_stats(sys_info, self._statistics_func)
        active_features = self._complete_features(
            sys_info, sys_output, external_stats=external_stats
        )
        samples_over_bucket, performance_over_bucket = self._bucketing_samples(
            sys_info, sys_output, active_features, scoring_stats=scoring_stats
        )
        overall_results = self.get_overall_performance(
            sys_info, sys_output, scoring_stats=scoring_stats
        )
        self._print_bucket_info(performance_over_bucket)
        sys_info.results = Result(
            overall=overall_results, fine_grained=performance_over_bucket
        )
        return sys_info
