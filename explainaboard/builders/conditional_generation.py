from typing import Any
from typing import Iterator, Dict, List

import numpy

# TODO(gneubig) we should try to remove this task-specific dependency with Datalab
from datalabs.operations.aggregate.summarization import summarization_aggregating
from datalabs.operations.featurize.plugins.summarization.sum_attribute import (
    SUMAttribute,
)
from datalabs.operations.featurize.summarization import get_oracle_summary
from tqdm import tqdm

from explainaboard.builders import ExplainaboardBuilder
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance
from explainaboard.utils.analysis import *

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
    def __init__(self):
        super().__init__()

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_source_length(self, existing_features: dict):
        return len(existing_features["source"].split(" "))

    def _get_reference_length(self, existing_features: dict):
        return len(existing_features["reference"].split(" "))

    def _get_hypothesis_length(self, existing_features: dict):
        return len(existing_features["hypothesis"].split(" "))

    # --- End feature functions

    # training set dependent features
    def _get_num_oov(self, existing_features: dict, statistics: Any):

        # exit()
        num_oov = 0

        for w in existing_features["source"].split(
            " "
        ):  # should this be normalized for the consistency with DataLab?
            if w not in statistics['vocab'].keys():
                num_oov += 1
        # print(num_oov)
        return num_oov

    # training set dependent features (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        fre_rank = 0

        for w in existing_features["source"].split(" "):
            if w not in statistics['vocab_rank'].keys():
                fre_rank += len(statistics['vocab_rank'])
            else:
                fre_rank += statistics['vocab_rank'][w]

        fre_rank = fre_rank * 1.0 / len(existing_features["source"].split(" "))
        return fre_rank

    def get_oracle(self, existing_features: dict, statistics: Any):
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
            statistics is not None
            and str(int(oracle_position)) in statistics['oracle_position_fre'].keys()
        ):
            oracle_position_fre = statistics['oracle_position_fre'][
                str(int(oracle_position))
            ]

        return {
            "oracle_position": oracle_position,
            "oracle_score": oracle_info["oracle_score"],
            "oracle_position_fre": oracle_position_fre,
        }

    # TODO(gneubig): can this be de-duplicated or is it specialized?
    def _complete_features(
        self, sys_info: SysOutputInfo, sys_output: List[dict], statistics=None
    ) -> List[str]:
        """
        This function is used to calculate features used for bucketing, such as sentence_length
        :return:
        """
        inputs = []
        for _id, feature_table in enumerate(sys_output):
            inputs.append(
                {
                    "source": feature_table["source"],
                    "references": [feature_table["reference"]],
                    "hypothesis": feature_table["hypothesis"],
                }
            )
            sys_output[_id] = feature_table

        self.score_dic = self._get_eaas_client().score(
            inputs,
            task="sum",
            metrics=sys_info.metric_names.copy(),
            lang="en",
            cal_attributes=False,
        )
        # print(self.score_dic["sample_level"][0].keys())

        # Get names of bucketing features
        # print(f"sys_info.features.get_bucket_features()\n {sys_info.features.get_bucket_features()}")

        # Get names of bucketing features
        oracle_feat_names = {"oracle_position", "oracle_score", "oracle_position_fre"}
        advanced_feat_names = set(summary_attribute.get_schema().keys())
        bucket_feature_funcs = {}
        for bucket_feature in sys_info.features.get_bucket_features():
            if bucket_feature in sys_info.features.keys() and (
                statistics is not None
                or not sys_info.features[bucket_feature].require_training_set
            ):
                if (
                    bucket_feature in oracle_feat_names
                    or bucket_feature in advanced_feat_names
                ):
                    bucket_feature_funcs[bucket_feature] = (None, False)
                else:
                    bucket_feature_funcs[bucket_feature] = (
                        self._get_feature_func(bucket_feature),
                        sys_info.features[bucket_feature].require_training_set,
                    )

        for _id, dict_sysout in enumerate(sys_output):
            dict_advanced_features = None
            oracle_feats = self.get_oracle(dict_sysout, statistics)
            # Get values of bucketing features
            for (
                bucket_key,
                (
                    bucket_func,
                    training_dependent,
                ),
            ) in bucket_feature_funcs.items():

                # TODO(gneubig): this logic seems complicated, can it be simplified?
                if bucket_key in advanced_feat_names:
                    if dict_advanced_features == None:
                        dict_advanced_features = summary_attribute.cal_attributes_each(
                            dict_sysout["source"], dict_sysout["reference"]
                        )
                    dict_sysout[bucket_key] = dict_advanced_features[bucket_key]
                elif bucket_key in oracle_feat_names:
                    dict_sysout[bucket_key] = oracle_feats[bucket_key]
                elif training_dependent:
                    dict_sysout[bucket_key] = bucket_func(dict_sysout, statistics)
                else:
                    dict_sysout[bucket_key] = bucket_func(dict_sysout)
        return list(bucket_feature_funcs.keys())

    # TODO(gneubig): should this be generalized or is it task specific?
    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
    ) -> Dict[str, Performance]:

        inputs = []  # noqa
        metrics = sys_info.metric_names  # noqa

        overall = {}
        for metric_name in sys_info.metric_names:

            overall_value = self.score_dic["corpus_level"]["corpus_" + metric_name]
            confidence_score_low = 0.0
            confidence_score_up = 0.0
            overall_performance = Performance(
                metric_name=metric_name,
                value=float(format(overall_value, '.4g')),
                confidence_score_low=float(format(confidence_score_low, '.4g')),
                confidence_score_up=float(format(confidence_score_up, '.4g')),
            )

            overall[metric_name] = overall_performance
        return overall

    # TODO(gneubig): this should be generalized
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
        :param samples_over_bucket: a dictionary mapping bucket interval names to lists of sample IDs for that bucket
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in samples_over_bucket.items():

            bucket_true_labels = []
            bucket_predicted_labels = []
            bucket_cases = []

            bucket_inputs = []
            dict_metric_to_values = {}

            for sample_id in sample_ids:
                sys_out = sys_output[sample_id]
                bucket_inputs.append(
                    {
                        "source": sys_out["source"],
                        "references": [sys_out["reference"]],
                        "hypothesis": sys_out["hypothesis"],
                    }
                )

                if sys_info.is_print_case:
                    # #bucket_case =  reference + "|||" + hypothesis
                    # bucket_case = {"source": (sample_id, ["source"]),
                    #                "references": (sample_id, ["references"]),
                    #                "hypothesis": (sample_id, ["hypothesis"])}
                    bucket_case = str(sample_id)
                    bucket_cases.append(bucket_case)

                for metric_name in sys_info.metric_names:
                    metric_value = self.score_dic["sample_level"][int(sample_id)][
                        metric_name
                    ]  # This would be modified later
                    if metric_name not in dict_metric_to_values.keys():
                        dict_metric_to_values[metric_name] = [metric_value]
                    else:
                        dict_metric_to_values[metric_name].append(metric_value)

            bucket_name_to_performance[bucket_interval] = []

            for metric_name in sys_info.metric_names:

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
                    value=bucket_value,
                    confidence_score_low=confidence_score_low,
                    confidence_score_up=confidence_score_up,
                    n_samples=len(dict_metric_to_values[metric_name]),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)  # noqa
