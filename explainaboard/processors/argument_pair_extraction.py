from __future__ import annotations

from collections.abc import Iterator

from datalabs import aggregating

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature_funcs import accumulate_vocab_from_samples
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.f1_score import APEF1ScoreConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.argument_pair_extraction)
class ArgumentPairExtractionProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.argument_pair_extraction

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        features = {
            "sentences": feature.Sequence(feature=feature.Value("string")),
            "true_tags": feature.Sequence(feature=feature.Value("string")),
            "pred_tags": feature.Sequence(feature=feature.Value("string")),
            "text_length": feature.Value(
                dtype="float",
                description="text length in tokens",
                func=lambda info, x, c: len(x['sentences']),
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
            APEF1ScoreConfig(
                name='APEF1Score',
                source_language=source_language,
                target_language=target_language,
            )
        ]

    def _get_true_label(self, data_point):
        """
        Get the true label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """
        return data_point["true_tags"]

    def _get_predicted_label(self, data_point):
        """
        Get the predicted label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["pred_tags"]

    @aggregating()
    def _statistics_func(self, samples: Iterator, sys_info: SysOutputInfo):
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples,
            lambda x: " ".join(x['sentences']),
            unwrap(sys_info.source_tokenizer),
        )

        return {'source_vocab': source_vocab, 'source_vocab_rank': source_vocab_rank}
