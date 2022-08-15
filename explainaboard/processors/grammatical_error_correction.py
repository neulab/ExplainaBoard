from __future__ import annotations

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.metrics.accuracy import SeqCorrectCountConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor


@register_processor(TaskType.grammatical_error_correction)
class GrammaticalErrorCorrection(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.grammatical_error_correction

    def default_analysis_levels(self) -> list[AnalysisLevel]:
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
