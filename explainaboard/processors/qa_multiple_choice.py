from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Sequence

from datalabs import aggregating

import explainaboard.analysis.analyses
from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.feature_funcs import (
    count_tokens,
    feat_freq_rank,
    feat_num_oov,
)
from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import AccuracyConfig, CorrectCountConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
import explainaboard.analysis.feature_funcs
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.qa_multiple_choice)
class QAMultipleChoiceProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.qa_multiple_choice

    @classmethod
    def default_analyses(cls) -> feature.Features:
        features = {
            "context": feature.Value("string"),
            "question": feature.Value("string"),
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
                description="context length in tokens",
                func=lambda info, x: count_tokens(info, x['context']),
            ),
            "question_length": feature.Value(
                dtype="float",
                description="context length in tokens",
                func=lambda info, x: count_tokens(info, x['question']),
            ),
            "answer_length": feature.Value(
                dtype="float",
                description="context length in tokens",
                func=lambda info, x: count_tokens(info, x['answers']['text'], side='target'),
            ),
            "num_oov": feature.Value(
                dtype="float",
                description="the number of out-of-vocabulary words in the context",
                require_training_set=True,
                func=lambda info, x, stat: feat_num_oov(info, x['context'], stat['vocab']),
            ),
            "fre_rank": feature.Value(
                dtype="float",
                description=(
                    "average rank of context words based on training set freq"
                ),
                require_training_set=True,
                func=lambda info, x, stat: feat_freq_rank(
                    info, x['context'], stat['vocab_rank']
                ),
            ),
        }
        continuous_features = [
            k for k, v in features.items() if ('float' in unwrap(v.dtype))
        ]
        analyses: Sequence[Analysis] = [
                                           BucketAnalysis(
                                               feature="true_label",
                                               method="discrete",
                                               number=15,
                                           )
                                       ] + [BucketAnalysis(x, method="continuous") for x in continuous_features]

        return [
            AnalysisLevel(
                name='example',
                features=features,
                metric_configs=cls.default_metrics(),
                analyses=analyses,
            )
        ]



        return feature.Features(
            {
                "num_oov": feature.Value(
                    dtype="float",
                    description="the number of out-of-vocabulary words",
                    is_bucket=True,
                    bucket_info=explainaboard.analysis.analyses.BucketAnalysis(
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
                    bucket_info=explainaboard.analysis.analyses.BucketAnalysis(
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

    def _get_question_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(unwrap(sys_info.source_tokenizer)(existing_features["question"]))

    def _get_answer_length(self, sys_info: SysOutputInfo, existing_features: dict):
        return len(
            unwrap(sys_info.target_tokenizer)(existing_features["answers"]["text"])
        )

    # training set dependent features
    def _get_num_oov(
        self, sys_info: SysOutputInfo, existing_features: dict, statistics: Any
    ):
        return explainaboard.analysis.feature_funcs.feat_num_oov(
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
        return explainaboard.analysis.feature_funcs.feat_freq_rank(
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
        return explainaboard.analysis.feature_funcs.accumulate_vocab_from_samples(
            samples, lambda x: x['context'], unwrap(sys_info.source_tokenizer)
        )
