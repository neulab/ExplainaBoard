from __future__ import annotations

from collections.abc import Iterator

from datalabs import aggregating

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature_funcs import (
    accumulate_vocab_from_samples,
    count_tokens,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.qa_table_text_hybrid import (
    ExactMatchHybridQAConfig,
    F1ScoreHybridQAConfig,
)
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.qa_table_text_hybrid)
class QATableTextHybridProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.qa_table_text_hybrid

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        features = {
            "question": feature.Value("string"),
            "context": feature.Sequence(feature=feature.Value("string")),
            "table": feature.Sequence(feature=feature.Value("string")),
            "true_answer": feature.Sequence(feature=feature.Value("string")),
            "predicted_answer": feature.Sequence(feature=feature.Value("string")),
            "answer_type": feature.Value("string"),
            "answer_scale": feature.Value("string"),
            "predicted_answer_scale": feature.Value("string"),
            "question_length": feature.Value(
                dtype="float",
                description="context length in tokens",
                func=lambda info, x, c: count_tokens(info, x['question']),
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
        return [
            ExactMatchHybridQAConfig(
                name='HybridQAExactMatch',
                source_language=source_language,
                target_language=target_language,
            ),
            F1ScoreHybridQAConfig(
                name='HybridQAF1',
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
        return {
            "true_answer": data_point["true_answer"],
            "answer_type": data_point["answer_type"],
            "answer_scale": data_point["answer_scale"],
        }

    def _get_predicted_label(self, data_point):
        """
        Get the predicted label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return {
            "predicted_answer": data_point["predicted_answer"],
            "predicted_answer_scale": data_point["predicted_answer_scale"],
        }

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples, lambda x: x['question'], unwrap(sys_info.source_tokenizer)
        )

        return {'source_vocab': source_vocab, 'source_vocab_rank': source_vocab_rank}
