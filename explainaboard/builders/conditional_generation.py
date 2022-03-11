from typing import Iterable, Optional
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance, Table
from explainaboard.builders import ExplainaboardBuilder
from explainaboard.utils.analysis import *  # noqa
from explainaboard.utils.eval_bucket import *  # noqa
import numpy
from tqdm import tqdm

from typing import Iterator, Dict, List
from datalabs import load_dataset

# TODO(gneubig) we should try to remove this task-specific dependency with Datalab
from datalabs.operations.aggregate.summarization import summarization_aggregating
from datalabs.operations.featurize.plugins.summarization.sum_attribute import (
    SUMAttribute,
)
from datalabs.operations.featurize.summarization import get_oracle_summary

# to calculate advanced features
summary_attribute = SUMAttribute()


# TODO(gneubig) this should be a member function
# TODO(gneubig) we should try to git rid of this task-specific decorator
# TODO(gneubig) should be conditional generation, not summarization
@summarization_aggregating(
    name="get_statistics",
    contributor="datalab",
    task="summarization",
    description="Calculate the overall statistics (e.g., density) of a given summarization dataset",
)
def get_statistics(samples: Iterator):
    """
    Input:
    samples: [{
     "text":
     "summary":
    }]
    Output:dict:
    """

    vocab = {}
    vocab_pruning = {}
    length_fre = {}
    oracle_position_fre = {}
    for sample in tqdm(samples):

        text, summary = sample["text"], sample["summary"]

        # oracle_position_fre
        oracle_info = get_oracle_summary.func(sample)
        index_of_oracles = [
            i for i, e in enumerate(oracle_info["oracle_labels"]) if e != 0
        ]
        oracle_position = str(int(numpy.mean(index_of_oracles)))

        if oracle_position not in oracle_position_fre.keys():
            oracle_position_fre[oracle_position] = 1
        else:
            oracle_position_fre[oracle_position] += 1

        # Vocabulary info
        for w in (text + summary).split(" "):
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1

    for k, v in vocab.items():  # pruning for the availability of database storage
        if v > 20:
            vocab_pruning[k] = v
        if len(vocab_pruning) > 100:
            break

    # the rank of each word based on its frequency
    sorted_dict = {
        key: rank
        for rank, key in enumerate(sorted(set(vocab_pruning.values()), reverse=True), 1)
    }
    vocab_rank = {k: sorted_dict[v] for k, v in vocab_pruning.items()}

    return {
        "vocab": vocab_pruning,
        "vocab_rank": vocab_rank,
        "oracle_position_fre": oracle_position_fre,
    }


class CondGenExplainaboardBuilder(ExplainaboardBuilder):
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

        # TODO(gneubig) to be deduplicated
        self._init_statistics(get_statistics)

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_source_length(self, existing_features: dict):
        return len(existing_features["source"].split(" "))

    def _get_reference_length(self, existing_features: dict):
        return len(existing_features["reference"].split(" "))

    def _get_hypothesis_length(self, existing_features: dict):
        return len(existing_features["hypothesis"].split(" "))

    # --- End feature functions

    # training set dependent features
    def _get_num_oov(self, existing_features: dict):

        # exit()
        num_oov = 0

        for w in existing_features["source"].split(
            " "
        ):  # should this be normalized for the consistency with DataLab?
            if w not in self.statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict):
        fre_rank = 0

        for w in existing_features["source"].split(" "):
            if w not in self.statistics['vocab_rank'].keys():
                fre_rank += len(self.statistics['vocab_rank'])
            else:
                fre_rank += self.statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(existing_features["source"].split(" "))
        return fre_rank

    def get_oracle(self, existing_features: dict):
        """
        oracle_info =
            {
            "source":src,
            "reference":ref,
            "oracle_summary":oracle,
            "oracle_labels":labels,
            "oracle_score":max_score
            }
        """

        sample = {
            "text": existing_features["source"],
            "summary": existing_features["reference"],
        }
        oracle_info = get_oracle_summary.func(sample)

        index_of_oracles = [
            i for i, e in enumerate(oracle_info["oracle_labels"]) if e != 0
        ]
        oracle_position = numpy.mean(index_of_oracles)

        oracle_position_fre = 0
        if (
            self.statistics != None
            and str(int(oracle_position))
            in self.statistics['oracle_position_fre'].keys()
        ):
            oracle_position_fre = self.statistics['oracle_position_fre'][
                str(int(oracle_position))
            ]

        return {
            "oracle_position": oracle_position,
            "oracle_score": oracle_info["oracle_score"],
            "oracle_position_fre": oracle_position_fre,
        }

    def _complete_feature(self):
        """
        This function is used to calculate features used for bucekting, such as sentence_length
        :param feature_table_iterator:
        :return:
        """
        inputs = []
        for _id, feature_table in enumerate(self._system_output):
            inputs.append(
                {
                    "source": feature_table["source"],
                    "references": [feature_table["reference"]],
                    "hypothesis": feature_table["hypothesis"],
                }
            )
            self._data[_id] = feature_table

        self.score_dic = self._get_eaas_client().score(
            inputs,
            task="sum",
            metrics=self._info.metric_names.copy(),
            lang="en",
            cal_attributes=False,
        )
        # print(self.score_dic["sample_level"][0].keys())

        # Get names of bucketing features
        # print(f"self._info.features.get_bucket_features()\n {self._info.features.get_bucket_features()}")
        bucket_features = self._info.features.get_bucket_features()

        for _id, dict_sysout in self._data.items():
            dict_advanced_features = None
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

                if bucket_feature in summary_attribute.get_schema().keys():
                    if dict_advanced_features == None:
                        dict_advanced_features = summary_attribute.cal_attributes_each(
                            dict_sysout["source"], dict_sysout["reference"]
                        )
                    feature_value = dict_advanced_features[bucket_feature]
                    dict_sysout[
                        bucket_feature
                    ] = feature_value  # !!!!!!!!!!!!!!!!!!!! string to float !!!!!
                elif bucket_feature in set(
                    ["oracle_position", "oracle_score", "oracle_position_fre"]
                ):
                    dict_sysout[bucket_feature] = self.get_oracle(dict_sysout)[
                        bucket_feature
                    ]
                else:
                    feature_value = self._get_feature_func(bucket_feature)(dict_sysout)
                    dict_sysout[bucket_feature] = feature_value

            self._data[_id] = dict_sysout
            yield _id, dict_sysout

    # TODO(gneubig): should this be generalized or is it task specific?
    def get_overall_performance(self):

        inputs = []  # noqa
        metrics = self._info.metric_names  # noqa

        for metric_name in self._info.metric_names:

            overall_value = self.score_dic["corpus_level"]["corpus_" + metric_name]
            confidence_score_low = 0.0
            confidence_score_up = 0.0
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

            bucket_true_labels = []  # noqa
            bucket_predicted_labels = []  # noqa
            bucket_cases = []

            bucket_inputs = []
            dict_metric_to_values = {}

            for sample_id in sample_ids:

                source = self._data[int(sample_id)]["source"]
                reference = self._data[int(sample_id)]["reference"]
                hypothesis = self._data[int(sample_id)]["hypothesis"]

                bucket_inputs.append(
                    {
                        "source": source,
                        "references": [reference],
                        "hypothesis": hypothesis,
                    }
                )

                if self._info.results.is_print_case:
                    # #bucket_case =  reference + "|||" + hypothesis
                    # bucket_case = {"source": (sample_id, ["source"]),
                    #                "references": (sample_id, ["references"]),
                    #                "hypothesis": (sample_id, ["hypothesis"])}
                    bucket_case = str(sample_id)
                    bucket_cases.append(bucket_case)

                for metric_name in self._info.metric_names:
                    metric_value = self.score_dic["sample_level"][int(sample_id)][
                        metric_name
                    ]  # This would be modified later
                    if metric_name not in dict_metric_to_values.keys():
                        dict_metric_to_values[metric_name] = [metric_value]
                    else:
                        dict_metric_to_values[metric_name].append(metric_value)

            bucket_name_to_performance[bucket_interval] = []

            for metric_name in self._info.metric_names:

                bucket_value = numpy.average(dict_metric_to_values[metric_name])
                confidence_score_low = 0.0
                confidence_score_up = 0.0

                # print(
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
                    n_samples=len(dict_metric_to_values[metric_name]),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)  # noqa
