from typing import Any
from typing import Iterator, Dict, List
from datalabs import aggregating

import explainaboard.utils.eval_basic_qa
import explainaboard.utils.feature_funcs
from explainaboard import feature
from explainaboard.info import SysOutputInfo, BucketPerformance, Performance
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
from explainaboard.utils.py_utils import sort_dict


@register_processor(TaskType.question_answering_extractive)
class QAExtractiveProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.question_answering_extractive

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "title": feature.Value("string"),
                "context": feature.Value("string"),
                "question": feature.Value("string"),
                "id": feature.Value("string"),
                "answers": feature.Sequence(feature.Value("string")),
                "predicted_answers": feature.Value("string"),
                "context_length": feature.Value(
                    dtype="float",
                    description="context length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "question_length": feature.Value(
                    dtype="float",
                    description="question length",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "answer_length": feature.Value(
                    dtype="float",
                    description="answer length",
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
                # "sim_context_question": feature.Value(dtype="float",
                #                                is_bucket=True,
                #                                bucket_info=feature.BucketInfo(
                #                                    method="bucket_attribute_specified_bucket_value",
                #                                    number=4,
                #                                    setting=()))
            }
        )

    @classmethod
    def default_metrics(cls) -> List[str]:
        return ["f1_score_qa", "exact_match_qa"]

    @aggregating()
    def _statistics_func(self, samples: Iterator):
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

        # TODO(gneubig): BEWARE THIS IS HACKY. This should use the same tokenizer as the processor.
        # tokenizer = SingleSpaceTokenizer()

        return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
            samples, lambda x: x['context'], self._tokenizer
        )

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["context"]))

    def _get_question_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["question"]))

    def _get_answer_length(self, existing_features: dict):
        if isinstance(existing_features["answers"]["text"], list):
            return len(self._tokenizer(existing_features["answers"]["text"][0]))
        else:
            return len(self._tokenizer(existing_features["answers"]["text"]))

    def _get_sim_context_question(self, existing_features: dict):

        references = existing_features["context"]
        hypothesis = existing_features["question"]

        res_json = self._get_eaas_client().bleu([[references]], [hypothesis], lang="en")
        return res_json["corpus_bleu"]

    # training set dependent features (could be merged for optimization?)
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features, statistics, lambda x: x['context'], self._tokenizer
        )

    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features, statistics, lambda x: x['context'], self._tokenizer
        )

    # --- End feature functions

    # TODO(gneubig): this can probably be generalized as well
    def get_overall_performance(
        self,
        sys_info: SysOutputInfo,
        sys_output: List[dict],
        scoring_stats: Any = None,
    ) -> Dict[str, Performance]:
        predicted_answers, true_answers = [], []

        for _id, feature_table in enumerate(sys_output):
            predicted_answers.append(feature_table["predicted_answers"]["text"])
            true_answers.append(feature_table["answers"]["text"])

        overall = {}
        for metric_name in sys_info.metric_names:
            # TODO(gneubig): is it necessary to have this as a separate interface than the other metrics?
            #                probably not. it'd be good to unify these.
            metric_func = getattr(explainaboard.utils.eval_basic_qa, metric_name)
            overall_value = metric_func(true_answers, predicted_answers)

            overall_value = overall_value
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
        scoring_stats: Any = None,
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

                true_label = sys_output[int(sample_id)]["answers"]["text"]
                if isinstance(true_label, list):
                    true_label = true_label[0]

                predicted_label = sys_output[int(sample_id)]["predicted_answers"][
                    "text"
                ]
                s_id = sys_output[int(sample_id)]["id"]

                # get a bucket of true/predicted labels
                bucket_true_labels.append(true_label)
                bucket_predicted_labels.append(predicted_label)
                # get a bucket of cases (e.g., errors)
                if sys_info.is_print_case:
                    if true_label != predicted_label:
                        # bucket_case = true_label[0] + "|||" + predicted_label + "|||" + sent
                        # bucket_case = {"true_answer": (sample_id, ["true_answers","text"]),
                        #                "predicted_answer": (sample_id, ["predicted_answer"]),
                        #                "question": (sample_id, ["question"])}
                        bucket_case = str(s_id)
                        bucket_cases.append(bucket_case)

            bucket_name_to_performance[bucket_interval] = []
            for metric_name in sys_info.metric_names:
                # TODO(gneubig): is it necessary to have this as a separate interface than the other metrics?
                #                probably not. it'd be good to unify these.
                metric_func = getattr(explainaboard.utils.eval_basic_qa, metric_name)
                bucket_value = metric_func(bucket_true_labels, bucket_predicted_labels)

                bucket_performance = BucketPerformance(
                    bucket_name=bucket_interval,
                    metric_name=metric_name,
                    value=bucket_value,
                    n_samples=len(bucket_true_labels),
                    bucket_samples=bucket_cases,
                )

                bucket_name_to_performance[bucket_interval].append(bucket_performance)

        return sort_dict(bucket_name_to_performance)


# @aggregating(
#     name="get_statistics",
#     contributor="datalab",
#     task="qa-extractive",
#     description="Calculate the overall statistics (e.g., average length) of "
#     "a given text classification dataset",
# )
# def get_statistics(samples: Iterator):
#     """
#     Input:
#     samples: [{
#      "id":str
#      "context":str
#      "question":str
#      "answers":Dict
#      "options"
#     }]
#     """
#
#     # TODO(gneubig): BEWARE THIS IS HACKY. This should use the same tokenizer as the processor.
#     tokenizer = SingleSpaceTokenizer()
#
#     return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
#         samples, lambda x: x['context'], tokenizer
#     )
