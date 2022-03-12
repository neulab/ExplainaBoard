from typing import Iterable, Optional
from explainaboard.info import SysOutputInfo, BucketPerformance, Performance, Table
from explainaboard.utils import analysis
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.eval_bucket import *  # noqa
from explainaboard.utils.analysis import *  # noqa
from explainaboard.metric import *  # noqa
from tqdm import tqdm
from typing import Iterator, Dict, List
from datalabs import load_dataset
from datalabs.operations.aggregate.qa_multiple_choice import (
    qa_multiple_choice_aggregating,
)


@qa_multiple_choice_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="qa-multiple-choice",
    description="Calculate the overall statistics (e.g., average length) of "
    "a given text classification dataset",
)
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "id":str
     "context":str
     "question":str
     "answers":Dict
     "options"
    }]
    """

    vocab = {}
    length_fre = {}
    for sample in tqdm(samples):
        context, answers, options = (
            sample["context"],
            sample["answers"],
            sample["options"],
        )

        # update vocabulary
        for w in context.split(" "):
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

    # the rank of each word based on its frequency
    sorted_dict = {
        key: rank
        for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
    }
    vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}

    # print(vocab)
    # print(vocab_rank)
    # exit()

    return {
        "vocab": vocab,
        "vocab_rank": vocab_rank,
    }


class QAMultipleChoiceExplainaboardBuilder(ExplainaboardBuilder):
    """
    Input: System Output file List[dict];  Metadata info
    Output: Analysis
    """

    def __init__(
        self,
        info: SysOutputInfo,
        system_output_object: Iterable[dict],
        feature_table: Optional[Table] = None,
        user_defined_feature_config=None,
    ):
        super().__init__(
            info, system_output_object, feature_table, user_defined_feature_config
        )

        # TODO(gneubig): this should be deduplicated
        # Calculate statistics of training set
        self.statistics = None
        if None != self._info.dataset_name:
            try:
                dataset = load_dataset(
                    self._info.dataset_name, self._info.sub_dataset_name
                )
                if (
                    len(dataset['train']._stat) == 0 or self._info.reload_stat == False
                ):  # calculate the statistics (_stat) when _stat is {} or `reload_stat` is False
                    new_train = dataset['train'].apply(get_statistics, mode="local")
                    self.statistics = new_train._stat
                else:
                    self.statistics = dataset["train"]._stat
            except FileNotFoundError as err:
                eprint(
                    "The dataset hasn't been supported by DataLab so no training set dependent features will be supported by ExplainaBoard."
                    "You can add the dataset by: https://github.com/ExpressAI/DataLab/blob/main/docs/SDK/add_new_datasets_into_sdk.md"
                )

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, existing_features: dict):
        return len(existing_features["context"].split(" "))

    def _get_question_length(self, existing_features: dict):
        return len(existing_features["question"].split(" "))

    def _get_answer_length(self, existing_features: dict):
        return len(existing_features["answers"]["text"].split(" "))

    # training set dependent features
    def _get_num_oov(self, existing_features: dict):
        num_oov = 0

        for w in existing_features["context"].split(" "):
            if w not in self.statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict):
        fre_rank = 0

        for w in existing_features["context"].split(" "):
            if w not in self.statistics['vocab_rank'].keys():
                fre_rank += len(self.statistics['vocab_rank'])
            else:
                fre_rank += self.statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(existing_features["context"].split(" "))
        return fre_rank

    # --- End feature functions

    def _complete_feature(self):
        """
        This function is used to calculate features used for bucekting, such as sentence_length
        :param feature_table_iterator:
        :return:
        """
        # Get names of bucketing features
        # print(f"self._info.features.get_bucket_features()\n {self._info.features.get_bucket_features()}")
        bucket_features = self._info.features.get_bucket_features()
        for _id, dict_sysout in tqdm(
            enumerate(self._system_output), desc="featurizing"
        ):
            # Get values of bucketing features
            for bucket_feature in bucket_features:

                # this is need due to `del self._info.features[bucket_feature]`
                if bucket_feature not in self._info.features.keys():
                    continue
                # If there is a training set dependent feature while no pre-computed statistics for it,
                # then skip bucketing along this feature
                if (
                    self._info.features[bucket_feature].require_training_set
                    and self.statistics == None
                ):
                    del self._info.features[bucket_feature]
                    continue

                feature_value = self._get_feature_func(bucket_feature)(dict_sysout)
                dict_sysout[bucket_feature] = feature_value
            # if self._data is None:
            #     self._data = {}
            self._data[_id] = dict_sysout
            yield _id, dict_sysout

    # TODO(gneubig): should this be generalized or is it task specific?
    def get_overall_performance(self):
        predicted_labels, true_labels = [], []

        for _id, feature_table in self._data.items():
            true_labels.append(feature_table["answers"]["option_index"])
            predicted_labels.append(feature_table["predicted_answers"]["option_index"])

        for metric_name in self._info.metric_names:
            one_metric = eval(metric_name)(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                is_print_confidence_interval=self._info.results.is_print_confidence_interval,
            )
            overall_value_json = one_metric.evaluate()

            overall_value = overall_value_json["value"]
            confidence_score_low = overall_value_json["confidence_score_low"]
            confidence_score_up = overall_value_json["confidence_score_up"]
            overall_performance = Performance(
                metric_name=metric_name,
                value=float(format(overall_value, '.4g')),
                confidence_score_low=float(format(confidence_score_low, '.4g')),
                confidence_score_up=float(format(confidence_score_up, '.4g')),
            )
            if self._info.results.overall is None:
                self._info.results.overall = {}
                self._info.results.overall[metric_name] = overall_performance
            else:
                self._info.results.overall[metric_name] = overall_performance

    def get_bucket_performance(self, feature_name: str):
        """
        This function defines how to get bucket-level performance w.r.t a given feature (e.g., sentence length)
        :param feature_name: the name of a feature, e.g., sentence length
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in self._samples_over_bucket[
            feature_name
        ].items():

            bucket_true_labels = []
            bucket_predicted_labels = []
            bucket_cases = []

            for sample_id in sample_ids:

                true_label = self._data[int(sample_id)]["answers"]["option_index"]
                predicted_label = self._data[int(sample_id)]["predicted_answers"][
                    "option_index"
                ]
                s_id = self._data[int(sample_id)]["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if self._info.results.is_print_case:
                    if true_label != predicted_label:
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in self._info.metric_names:
                one_metric = eval(metric_name)(
                    true_labels=bucket_true_labels,
                    predicted_labels=bucket_predicted_labels,
                    is_print_confidence_interval=self._info.results.is_print_confidence_interval,
                )
                bucket_value_json = one_metric.evaluate()

                bucket_value = bucket_value_json["value"]
                confidence_score_low = bucket_value_json["confidence_score_low"]
                confidence_score_up = bucket_value_json["confidence_score_up"]

                # print(f"name:\t {one_metric._name} \n"
                #       f"value:\t {bucket_value}\n"
                #       f"confidence low\t {confidence_score_low}\n"
                #       f"confidence up \t {confidence_score_up}\n"
                #       f"---------------------------------")

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=format(bucket_value, '.4g'),
                    confidence_score_low=format(confidence_score_low, '.4g'),
                    confidence_score_up=format(confidence_score_up, '.4g'),
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)  # noqa
