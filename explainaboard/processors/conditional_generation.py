from typing import Any
from typing import Iterator, Dict, List

import numpy
from tqdm import tqdm


from datalabs import aggregating
import explainaboard.utils.feature_funcs
from explainaboard import feature
from explainaboard.info import SysOutputInfo, Performance, BucketPerformance
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from explainaboard.utils.py_utils import sort_dict
from explainaboard.utils.tokenizer import SingleSpaceTokenizer


@register_processor(TaskType.conditional_generation)
class ConditionalGenerationProcessor(Processor):
    _task_type = TaskType.conditional_generation
    _default_metrics = ["rouge1", "rouge2", "rougeL", "bleu"]

    _features = feature.Features(
        {
            "source": feature.Value("string"),
            "reference": feature.Value("string"),
            "hypothesis": feature.Value("string"),
            "source_length": feature.Value(
                dtype="float",
                description="the length of source document",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "reference_length": feature.Value(
                dtype="float",
                description="the length of gold summary",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "hypothesis_length": feature.Value(
                dtype="float",
                description="the length of gold summary",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
            ),
            "num_oov": feature.Value(
                dtype="float",
                description="the number of out-of-vocabulary words",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
                require_training_set=True,
            ),
            "fre_rank": feature.Value(
                dtype="float",
                description="the average rank of each work based on its frequency in training set",
                is_bucket=True,
                bucket_info=feature.BucketInfo(
                    method="bucket_attribute_specified_bucket_value",
                    number=4,
                    setting=(),
                ),
                require_training_set=True,
            ),
        }
    )

    def __init__(self):
        super().__init__()
        self._statistics_func = get_statistics

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_source_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["source"]))

    def _get_reference_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["reference"]))

    def _get_hypothesis_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["hypothesis"]))

    # training set dependent features (could be merged for optimization?)
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features, statistics, lambda x: x['source']
        )

    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features, statistics, lambda x: x['source']
        )

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

        request_id = self._get_eaas_client().async_score(
            inputs,
            task="sum",  # TODO(pengfei): this should be generalized
            metrics=sys_info.metric_names.copy(),
            lang="en",
            cal_attributes=False,
        )

        # Get names of bucketing features
        # print(f"sys_info.features.get_bucket_features()\n {sys_info.features.get_bucket_features()}")

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

        for _id, dict_sysout in enumerate(sys_output):

            # oracle_feats = self.get_oracle(dict_sysout, statistics)
            # Get values of bucketing features
            for (
                bucket_key,
                (
                    bucket_func,
                    training_dependent,
                ),
            ) in bucket_feature_funcs.items():
                # TODO(pengfei): should check the feature value type
                if training_dependent:
                    dict_sysout[bucket_key] = bucket_func(dict_sysout, statistics)
                else:
                    dict_sysout[bucket_key] = bucket_func(dict_sysout)
                    # print(dict_sysout[bucket_key])

        self.score_dict = self._eaas_client.wait_and_get_result(request_id)
        return list(bucket_feature_funcs.keys())

    # TODO(gneubig): should this be generalized or is it task specific?
    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
    ) -> Dict[str, Performance]:

        overall = {}
        for metric_name in sys_info.metric_names:

            overall_value = self.score_dict["corpus_level"]["corpus_" + metric_name]
            overall_performance = Performance(
                metric_name=metric_name,
                value=overall_value,
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
        :param samples_over_bucket: a dictionary mapping bucket interval names to sample IDs for that bucket
        :return: bucket_name_to_performance: a dictionary that maps bucket names to bucket performance
        """

        bucket_name_to_performance = {}
        for bucket_interval, sample_ids in samples_over_bucket.items():

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
                    metric_value = self.score_dict["sample_level"][int(sample_id)][
                        metric_name
                    ]  # This would be modified later
                    if metric_name not in dict_metric_to_values.keys():
                        dict_metric_to_values[metric_name] = [metric_value]
                    else:
                        dict_metric_to_values[metric_name].append(metric_value)

            bucket_name_to_performance[bucket_interval] = []

            for metric_name in sys_info.metric_names:

                bucket_value = numpy.average(dict_metric_to_values[metric_name])

                # print(
                #       f"value:\t {bucket_value}\n"
                #       f"confidence low\t {confidence_score_low}\n"
                #       f"confidence up \t {confidence_score_high}\n"
                #       f"---------------------------------")

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=bucket_value,
                    n_samples=len(dict_metric_to_values[metric_name]),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)


# @register_processor(TaskType.summarization)
# class SummarizationProcessor(ConditionalGenerationProcessor):
#     _task_type = TaskType.summarization
#     _default_metrics = ["rouge1", "rouge2", "rougeL"]


# TODO(gneubig) should be conditional generation, not summarization
# Aggregate training set statistics
@aggregating(
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

    # TODO(gneubig): BEWARE THIS IS HACKY. This should use the same tokenizer as the processor.
    tokenizer = SingleSpaceTokenizer()

    vocab = {}
    vocab_pruning = {}

    for sample in tqdm(samples):

        text, summary = sample["text"], sample["summary"]

        # # oracle_position_fre
        # oracle_info = get_oracle_summary.func(sample)
        # index_of_oracles = [
        #     i for i, e in enumerate(oracle_info["oracle_labels"]) if e != 0
        # ]
        # oracle_position = str(int(numpy.mean(index_of_oracles)))
        #
        # if oracle_position not in oracle_position_fre.keys():
        #     oracle_position_fre[oracle_position] = 1
        # else:
        #     oracle_position_fre[oracle_position] += 1

        # Vocabulary info
        for w in tokenizer(text + summary):
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
        # "oracle_position_fre": oracle_position_fre,
    }
