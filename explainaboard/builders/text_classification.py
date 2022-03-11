from typing import Optional, Iterable
from explainaboard.info import SysOutputInfo, BucketPerformance, Table
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.analysis import *  # noqa
from explainaboard.utils.eval_bucket import *  # noqa
from explainaboard.metric import Accuracy  # noqa
from explainaboard.metric import F1score  # noqa
from tqdm import tqdm
from explainaboard.utils.feature_funcs import *  # noqa
from explainaboard.utils.spacy_loader import spacy_loader
from typing import Iterator, Dict, List
from datalabs import load_dataset
from datalabs.operations.aggregate.text_classification import (
    text_classification_aggregating,
)
import requests
import json
from explainaboard.utils.db_api import *


@text_classification_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="text-classification",
    description="Calculate the overall statistics (e.g., average length) of "
    "a given text classification dataset",
)
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text":
     "label":
    }]
    """

    vocab = {}
    length_fre = {}
    for sample in tqdm(samples):
        text, label = sample["text"], sample["label"]
        length = len(text.split(" "))

        if length in length_fre.keys():
            length_fre[length] += 1
        else:
            length_fre[length] = 1

        # update vocabulary
        for w in text.split(" "):
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

    for k, v in length_fre.items():
        length_fre[k] = length_fre[k] * 1.0 / len(samples)

    return {"vocab": vocab, "vocab_rank": vocab_rank, "length_fre": length_fre}


class TCExplainaboardBuilder(ExplainaboardBuilder):
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
        self._init_statistics(get_statistics)

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_sentence_length(self, existing_features: dict):
        return len(existing_features["text"].split(" "))

    def _get_token_number(self, existing_feature: dict):
        return len(existing_feature["text"])

    def _get_entity_number(self, existing_feature: dict):
        return len(
            spacy_loader.get_model("en_core_web_sm")(existing_feature["text"]).ents
        )

    def _get_label(self, existing_feature: dict):
        return existing_feature["true_label"]

    def _get_basic_words(self, existing_feature: dict):
        return get_basic_words(existing_feature["text"])  # noqa

    def _get_lexical_richness(self, existing_feature: dict):
        return get_lexical_richness(existing_feature["text"])  # noqa

    # training set dependent features
    def _get_num_oov(self, existing_features: dict):
        num_oov = 0

        for w in existing_features["text"].split(" "):
            if w not in self.statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict):
        fre_rank = 0

        for w in existing_features["text"].split(" "):
            if w not in self.statistics['vocab_rank'].keys():
                fre_rank += len(self.statistics['vocab_rank'])
            else:
                fre_rank += self.statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(existing_features["text"].split(" "))
        return fre_rank

    # training set dependent features
    def _get_length_fre(self, existing_features: dict):
        length_fre = 0
        length = len(existing_features["text"].split(" "))

        if length in self.statistics['length_fre'].keys():
            length_fre = self.statistics['length_fre'][length]

        return length_fre

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

                true_label = self._data[int(sample_id)]["true_label"]
                predicted_label = self._data[int(sample_id)]["predicted_label"]
                sent = self._data[int(sample_id)]["text"]  # noqa
                s_id = self._data[int(sample_id)]["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if self._info.results.is_print_case:
                    if true_label != predicted_label:
                        # bucket_case = true_label + "|||" + predicted_label + "|||" + sent
                        # bucket_case = {"true_label":(s_id,["true_label"]),
                        #                "predicted_label":(s_id,["predicted_label"]),
                        #                "text":(s_id,["text"])}
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
