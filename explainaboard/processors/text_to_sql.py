from __future__ import annotations

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import Analysis, AnalysisLevel
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.sql_em_ex import SQLEmConfig, SQLExConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor


@register_processor(TaskType.text_to_sql)
class TextToSQLProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.text_classification

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        features: dict[str, FeatureType] = {
            "question": feature.Value(
                dtype="string",
                description="the input question",
            ),
            "true_sql": feature.Value(
                dtype="string",
                description="the true sql",
            ),
            "predicted_sql": feature.Value(
                dtype="string",
                description="the predicted sql",
            ),
            "db_id": feature.Value(
                dtype="string",
                description="the database id",
            ),
            "text_length": feature.Value(
                dtype="float",
                description="text length in tokens",
                func=lambda info, x, c: count_tokens(info, x["question"]),
            ),
        }

        return [
            AnalysisLevel(
                name="example",
                features=features,
                metric_configs=self.default_metrics(),
            )
        ]

    def default_analyses(self) -> list[Analysis]:
        return self.continuous_feature_analyses()

    @classmethod
    def default_metrics(
        cls, level="example", source_language=None, target_language=None
    ) -> list[MetricConfig]:
        return [
            SQLEmConfig(
                name="Em",
                source_language=source_language,
                target_language=target_language,
            ),
            SQLExConfig(
                name="Ex",
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
        return data_point["true_sql"]

    def _get_predicted_label(self, data_point):
        """
        Get the predicted label from a data point. Overloaded from parent class.
        :param data_point: the data point under consideration
        :return: the predicted label for the output
        """
        return data_point["predicted_sql"]
