from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datalabs import aggregating

from explainaboard import feature
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.tasks import TaskType
import explainaboard.utils.feature_funcs
from explainaboard.utils.tokenizer import SingleSpaceTokenizer


@register_processor(TaskType.qa_multiple_choice)
class QAMultipleChoiceProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.qa_multiple_choice

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "context": feature.Value("string"),
                "question": feature.Value("string"),
                "options": feature.Sequence(feature.Value("string")),
                "answers": {
                    "text": feature.Value("string"),
                    "option_index": feature.Value("int32"),
                },
                "context_length": feature.Value(
                    dtype="float",
                    description="the length of context",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "question_length": feature.Value(
                    dtype="float",
                    description="the length of question",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "answer_length": feature.Value(
                    dtype="float",
                    description="the length of answer",
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
                    description=(
                        "the average rank of each work based on its frequency in "
                        "training set"
                    ),
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

    @classmethod
    def default_metrics(cls) -> list[str]:
        return ["Accuracy"]

    def __init__(self):
        super().__init__()
        # self._statistics_func = get_statistics

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["context"]))

    def _get_question_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["question"]))

    def _get_answer_length(self, existing_features: dict):
        return len(self._tokenizer(existing_features["answers"]["text"]))

    # training set dependent features
    def _get_num_oov(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features, statistics, lambda x: x['context'], self._tokenizer
        )

    # training set dependent features
    # (this could be merged into the above one for further optimization)
    def _get_fre_rank(self, existing_features: dict, statistics: Any):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features, statistics, lambda x: x['context'], self._tokenizer
        )

    # --- End feature functions

    def _get_true_label(self, data_point):
        """
        Get the true label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """
        return data_point["answers"]["option_index"]

    def _get_predicted_label(self, data_point):
        """
        Get the predicted label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["predicted_answers"]["option_index"]


@aggregating(
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

    # TODO(gneubig):
    # BEWARE THIS IS HACKY. This should use the same tokenizer as the processor.
    tokenizer = SingleSpaceTokenizer()

    return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
        samples, lambda x: x['context'], tokenizer
    )
