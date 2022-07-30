from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datalabs import aggregating

from explainaboard import feature, TaskType
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.extractive_qa import ExactMatchQAConfig, F1ScoreQAConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
import explainaboard.utils.feature_funcs
from explainaboard.utils.feature_funcs import accumulate_vocab_from_samples
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.qa_open_domain)
class QAOpenDomainProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.qa_open_domain

    @classmethod
    def default_features(cls) -> feature.Features:
        return feature.Features(
            {
                "question": feature.Value("string"),
                "question_types": feature.Sequence(feature=feature.Value("string")),
                "answers": feature.Sequence(feature=feature.Value("string")),
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
                "question_type": feature.Value(
                    dtype="string",
                    description="question type",
                    is_bucket=True,
                    bucket_info=feature.BucketInfo(
                        method="bucket_attribute_discrete_value", number=4, setting=1
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
            ExactMatchQAConfig(
                name='ExactMatch',
                source_language=source_language,
                target_language=target_language,
            ),
            F1ScoreQAConfig(
                name='F1',
                source_language=source_language,
                target_language=target_language,
            ),
        ]

    def __init__(self):
        super().__init__()

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()
    def _get_question_type(self, sys_info: SysOutputInfo, existing_feature: dict):
        return " ".join(existing_feature["question_types"])

    def _get_question_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["question"]))

    def _get_answer_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.target_tokenizer)(existing_features["answers"][0]))

    # training set dependent features
    def _get_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_num_oov(
            existing_features,
            statistics['source_vocab'],
            lambda x: x['question'],
            unwrap(sys_info.source_tokenizer),
        )

    # training set dependent features
    # (this could be merged into the above one for further optimization)
    def _get_fre_rank(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.utils.feature_funcs.feat_freq_rank(
            existing_features,
            statistics['source_vocab_rank'],
            lambda x: x['question'],
            unwrap(sys_info.source_tokenizer),
        )

    # --- End feature functions

    def _get_true_label(self, data_point):
        """
        Get the true label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """
        return data_point["answers"]

    def _get_predicted_label(self, data_point):
        """
        Get the predicted label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["predicted_answer"]

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x['question'], unwrap(sys_info.source_tokenizer)
        )

        return {'source_vocab': source_vocab, 'source_vocab_rank': source_vocab_rank}