from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datalabs import aggregating

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metric import AccuracyConfig, CorrectCountConfig, MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
import explainaboard.utils.feature_funcs
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.cloze_mutiple_choice)
class ClozeMultipleChoiceProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.cloze_mutiple_choice

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "context": feature.Value("string"),
                "question_mark": feature.Value("string"),
                "options": feature.Sequence(feature=feature.Value("string")),
                "answers": feature.Sequence(
                    feature=feature.Dict(
                        feature={
                            "text": feature.Value("string"),
                            "option_index": feature.Value("int32"),
                        }
                    )
                ),
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
                "relative_blank_position": feature.Value(
                    dtype="float",
                    description="the relative position of blank (question mark)"
                    " in the whole context",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_specified_bucket_value",
                        number=4,
                        setting=(),
                    ),
                ),
                "absolute_blank_position": feature.Value(
                    dtype="float",
                    description="the absolute position of blank (question mark)"
                    " in the whole context",
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
                        "the average rank of each word based on its frequency in "
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
    def default_metrics(
        cls, source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            AccuracyConfig(
                name='Accuracy',
                source_language=source_language,
                target_language=target_language,
            ),
            CorrectCountConfig(
                name='CorrectCount',
                source_language=source_language,
                target_language=target_language,
            ),
        ]

    def __init__(self):
        super().__init__()

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_context_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["context"]))

    def _get_relative_blank_position(
        self, sys_info: SysOutputInfo, existing_features: dict
    ):
        source_tokens = unwrap(sys_info.source_tokenizer)(
            existing_features["context"]
        ).strs
        if existing_features["question_mark"] not in source_tokens:
            return 0
        else:
            return (
                source_tokens.index(existing_features["question_mark"])
                * 1.0
                / len(source_tokens)
            )

    def _get_absolute_blank_position(
        self, sys_info: SysOutputInfo, existing_features: dict
    ):
        source_tokens = unwrap(sys_info.source_tokenizer)(
            existing_features["context"]
        ).strs
        if existing_features["question_mark"] not in source_tokens:
            return 0
        else:
            return source_tokens.index(existing_features["question_mark"])

    def _get_answer_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(
            unwrap(sys_info.target_tokenizer)(existing_features["answers"]["text"])
        )

    # training set dependent features
    def _get_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features,
            statistics,
            lambda x: x['context'],
            unwrap(sys_info.source_tokenizer),
        )

    # training set dependent features
    # (this could be merged into the above one for further optimization)
    def _get_fre_rank(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features,
            statistics,
            lambda x: x['context'],
            unwrap(sys_info.source_tokenizer),
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

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        return explainaboard.utils.feature_funcs.accumulate_vocab_from_samples(
            samples, lambda x: x['context'], unwrap(sys_info.source_tokenizer)
        )
