from __future__ import annotations

from typing import cast, List

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel, BucketAnalysis
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.metrics.accuracy import SeqCorrectCountConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.grammatical_error_correction)
class GrammaticalErrorCorrection(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.grammatical_error_correction

    def default_analyses(self) -> list[AnalysisLevel]:
        features: dict[str, FeatureType] = feature.Features(
            {
                "text": feature.Value("string"),
                "edits": feature.Dict(
                    feature={
                        "start_idx": feature.Sequence(feature=feature.Value("int32")),
                        "end_idx": feature.Sequence(feature=feature.Value("int32")),
                        "corrections": feature.Sequence(
                            feature=feature.Sequence(feature=feature.Value("string"))
                        ),
                    }
                ),
                "text_length": feature.Value(
                    dtype="float",
                    description="length of the text",
                    func=lambda info, x, c: count_tokens(info, x['text']),
                ),
            }
        )
        continuous_features = [
            k for k, v in features.items() if ('float' in unwrap(v.dtype))
        ]
        analyses: list[BucketAnalysis] = [
            BucketAnalysis(
                description=features[x].description, feature=x, method="continuous"
            )
            for x in continuous_features
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
        return [SeqCorrectCountConfig(name='SeqCorrectCount')]

    # --- Feature functions accessible by ExplainaboardBuilder._get_feature_func()

    def _get_true_label(self, data_point):
        """
        Get the true label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the true label for the output
        """

        return data_point["edits"]

    def _get_predicted_label(self, data_point):
        """
        Get the predicted label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["predicted_edits"]
