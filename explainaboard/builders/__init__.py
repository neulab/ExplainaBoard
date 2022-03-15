from typing import Callable, List, Tuple, Dict, Iterator, Any
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance, Result
import explainaboard.metric
from tqdm import tqdm
from eaas import Config, Client
from datalabs import load_dataset
import explainaboard.utils.bucketing
from explainaboard.utils.analysis import (
    eprint,
    print_dict,
    sort_dict,
)
from explainaboard.utils.db_api import *


class ExplainaboardBuilder:
    def __init__(self):
        # Things to use only if necessary
        self._eaas_config = None
        self._eaas_client = None
        self._statistics_func = None

    def _init_statistics(self, sys_info: SysOutputInfo, statistics_func: Callable):
        """Take in information about the system outputs and a statistic calculating function and return a dictionary
        of statistics.

        :param sys_info: Information about the system outputs
        :param statistics_func: The function used to get the statistics
        :return: Statistics from, usually, the training set that are used to calculate other features
        """
        statistics = None
        if sys_info.dataset_name is not None:
            dataset_name = sys_info.dataset_name
            split_name = "train"
            sub_dataset = (
                None
                if sys_info.sub_dataset_name == "default"
                else sys_info.sub_dataset_name
            )
            try:
                # read statistics from db
                response = read_statistics_from_db(dataset_name, sub_dataset)
                message = json.loads(response.text.replace("null", ""))["message"]
                eprint(message)
                if message == "success" and sys_info.reload_stat:
                    statistics = json.loads(response.content)['content']
                elif (
                    message == "success"
                    and not sys_info.reload_stat
                    or message
                    == "the dataset does not include the information of _stat"
                ):
                    dataset = load_dataset(
                        sys_info.dataset_name, sys_info.sub_dataset_name
                    )
                    if (
                        len(dataset[split_name]._stat) == 0 or not sys_info.reload_stat
                    ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                        new_train = dataset[split_name].apply(
                            statistics_func, mode="local"
                        )

                        statistics = new_train._stat
                        # self.statistics = dataset['train']._stat
                        # write statistics to db
                        eprint("saving to database")
                        response = write_statistics_from_db(
                            dataset_name, sub_dataset, content=statistics
                        )
                        eprint(response.content)
                else:  # dataset does not exist
                    eprint(
                        "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard." # noqa
                        "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md" # noqa
                    )
            except FileNotFoundError:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard." # noqa
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md" # noqa
                )
        return statistics

    def _get_feature_func(self, func_name: str):
        return getattr(self, f'_get_{func_name}')

    def _get_eaas_client(self):
        if not self._eaas_client:
            self._eaas_config = Config()
            self._eaas_client = Client()
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
        self, sys_info: SysOutputInfo, sys_output: List[dict], statistics=None
    ) -> List[str]:
        """
        This function takes in meta-data about system outputs, system outputs, and a few other optional pieces of
        information, then calculates feature functions and modifies `sys_output` to add these feature values

        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param statistics: Training set statistics that are used to calculate training set specific features
        :return: The features that are active (e.g. skipping training set features when no training set available)
        """
        # Get names of bucketing features
        bucket_feature_funcs = {}
        for bucket_feature in sys_info.features.get_bucket_features():
            if bucket_feature in sys_info.features.keys() and (
                statistics is not None
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
                dict_sysout[bucket_key] = (
                    bucket_func(dict_sysout, statistics)
                    if training_dependent
                    else bucket_func(dict_sysout)
                )
        return list(bucket_feature_funcs.keys())

    def _bucketing_samples(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        active_features: List[str],
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
                sys_info, sys_output, samples_over_bucket[feature_name]
            )

        return samples_over_bucket, performances_over_bucket

    def get_bucket_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        samples_over_bucket: Dict[str, List[int]],
    ) -> Dict[str, List[BucketPerformance]]:
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample IDs for that bucket
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
                    confidence_score_up=metric_result["confidence_score_up"],
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)

    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
    ) -> Dict[str, Performance]:
        """
        Get the overall performance according to metrics
        :param sys_info: Information about the system output
        :param sys_output: The system output itself
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
                confidence_score_up=metric_result["confidence_score_up"],
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

    def run(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
    ) -> Result:
        statistics = self._init_statistics(sys_info, self._statistics_func)
        active_features = self._complete_features(
            sys_info, sys_output, statistics=statistics
        )
        samples_over_bucket, performance_over_bucket = self._bucketing_samples(
            sys_info, sys_output, active_features
        )
        overall_results = self.get_overall_performance(sys_info, sys_output)
        self._print_bucket_info(performance_over_bucket)
        result = Result(overall=overall_results, fine_grained=performance_over_bucket)
        return result

    # ------ Below are utility functions for feature calculation -------
    @staticmethod
    def accumulate_vocab_from_samples(samples: Iterator, text_from_sample: Callable):
        vocab = {}
        for sample in tqdm(samples):
            for w in text_from_sample(sample).split(" "):
                vocab[w] = vocab.get(w, 0)+1
        # the rank of each word based on its frequency
        sorted_dict = {
            key: rank
            for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
        }
        vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}
        return {
            "vocab": vocab,
            "vocab_rank": vocab_rank,
        }

    @staticmethod
    def feat_freq_rank(existing_features: dict, statistics: Any, text_from_sample: Callable):
        fre_rank = 0

        tokens = text_from_sample(existing_features).split(" ")
        for w in tokens:
            if w not in statistics['vocab_rank']:
                fre_rank += len(statistics['vocab_rank'])
            else:
                fre_rank += statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(tokens)
        return fre_rank

    @staticmethod
    def feat_num_oov(existing_features: dict, statistics: Any, text_from_sample: Callable):
        num_oov = 0
        for w in text_from_sample(existing_features).split(" "):
            if w not in statistics['vocab'].keys():
                num_oov += 1
        return num_oov
