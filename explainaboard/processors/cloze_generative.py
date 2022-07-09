from __future__ import annotations

from collections.abc import Iterator
from typing import cast, List

from datalabs import aggregating

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import (
    absolute_position,
    accumulate_vocab_from_samples,
    count_tokens,
    feat_freq_rank,
    feat_num_oov,
    relative_position,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import CorrectCountConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.cloze_generative)
class ClozeGenerativeProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.cloze_generative

    def default_analyses(self) -> list[AnalysisLevel]:
        features: dict[str, FeatureType] = {
            "context": feature.Value("string"),
            "question_mark": feature.Value("string"),
            "hint": feature.Value("string"),
            "answers": feature.Value("string"),
            "context_length": feature.Value(
                dtype="float",
                description="the length of context",
                func=lambda info, x, c: count_tokens(info, x['context']),
            ),
            "relative_blank_position": feature.Value(
                dtype="float",
                description="the relative position of blank (question mark)"
                " in the whole context",
                func=lambda info, x, c: relative_position(
                    info, x['context'], x['question_mark']
                ),
            ),
            "absolute_blank_position": feature.Value(
                dtype="float",
                description="the absolute position of blank (question mark)"
                " in the whole context",
                func=lambda info, x, c: absolute_position(
                    info, x['context'], x['question_mark']
                ),
            ),
            "answer_length": feature.Value(
                dtype="float",
                description="the length of answer",
                func=lambda info, x, c: count_tokens(info, x['answers']),
            ),
            "num_oov": feature.Value(
                dtype="float",
                description="the number of out-of-vocabulary words",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x['text'], stat['source_vocab']
                ),
            ),
            "fre_rank": feature.Value(
                dtype="float",
                description=(
                    "the average rank of each word based on its frequency in "
                    "training set"
                ),
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x['text'], stat['source_vocab_rank']
                ),
            ),
        }
        continuous_features = [
            k for k, v in features.items() if ('float' in unwrap(v.dtype))
        ]
        analyses: list[BucketAnalysis] = [
            BucketAnalysis(x, method="continuous") for x in continuous_features
        ]

        return [
            AnalysisLevel(
                name='example',
                features=features,
                metric_configs=self.default_metrics(),
                analyses=cast(List[Analysis], analyses),
            )
        ]

    @classmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            CorrectCountConfig(
                name='CorrectCount',
                source_language=source_language,
                target_language=target_language,
            ),
        ]

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
        return data_point["predicted_answers"][0]

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x['context'], unwrap(sys_info.source_tokenizer)
        )

        return {'source_vocab': source_vocab, 'source_vocab_rank': source_vocab_rank}
