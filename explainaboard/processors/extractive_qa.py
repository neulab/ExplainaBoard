from __future__ import annotations

from collections.abc import Iterator

from datalabs import aggregating

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature_funcs import (
    accumulate_vocab_from_samples,
    count_tokens,
    feat_freq_rank,
    feat_num_oov,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.extractive_qa import ExactMatchQAConfig, F1ScoreQAConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.qa_extractive)
class QAExtractiveProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.qa_extractive

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        features = {
            "context": feature.Value("string"),
            "question": feature.Value("string"),
            "id": feature.Value("string"),
            "answers": feature.Sequence(feature=feature.Value("string")),
            "predicted_answers": feature.Value("string"),
            "context_length": feature.Value(
                dtype="float",
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(info, x['context']),
            ),
            "question_length": feature.Value(
                dtype="float",
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(info, x['question']),
            ),
            "answer_length": feature.Value(
                dtype="float",
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(
                    info,
                    x['answers']['text'][0]
                    if isinstance(x["answers"]["text"], list)
                    else x['answers']['text'],
                    side='target',
                ),
            ),
            "num_oov": feature.Value(
                dtype="float",
                description="the number of out-of-vocabulary words in the context",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x['context'], stat['source_vocab']
                ),
            ),
            "fre_rank": feature.Value(
                dtype="float",
                description=(
                    "average rank of context words based on training set freq"
                ),
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x['context'], stat['source_vocab_rank']
                ),
            ),
        }

        return [
            AnalysisLevel(
                name='example',
                features=features,
                metric_configs=self.default_metrics(),
            )
        ]

    def default_analyses(self) -> list[Analysis]:
        return self.continuous_feature_analyses()

    @classmethod
    def default_metrics(
        cls, level='example', source_language=None, target_language=None
    ) -> list[MetricConfig]:
        if source_language != target_language:
            raise ValueError(
                'Source and target language must be equal for extractive '
                f'QA, but got {source_language} and {target_language}'
            )
        return [
            F1ScoreQAConfig(
                name='F1',
                source_language=source_language,
                target_language=target_language,
            ),
            ExactMatchQAConfig(
                name='ExactMatch',
                source_language=source_language,
                target_language=target_language,
            ),
        ]

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x['context'], unwrap(sys_info.source_tokenizer)
        )

        return {'source_vocab': source_vocab, 'source_vocab_rank': source_vocab_rank}

    def _get_true_label(self, data_point: dict):
        return data_point["answers"]["text"]

    def _get_predicted_label(self, data_point: dict):
        return data_point["predicted_answers"]["text"]
