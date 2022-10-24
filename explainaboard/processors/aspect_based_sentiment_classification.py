"""A processor for the aspect based sentiment classification task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import (
    Analysis,
    AnalysisLevel,
    BucketAnalysis,
    CalibrationAnalysis,
    ComboCountAnalysis,
)
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.utils.spacy_loader import get_named_entities


class AspectBasedSentimentClassificationProcessor(Processor):
    """A processor for the aspect based sentiment classification task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.aspect_based_sentiment_classification

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features: dict[str, FeatureType] = {
            "aspect": feature.Value(
                dtype=feature.DataType.STRING,
                description="the aspect to analyze",
            ),
            "text": feature.Value(
                dtype=feature.DataType.STRING,
                description="the text regarding the aspect",
            ),
            "true_label": feature.Value(
                dtype=feature.DataType.STRING,
                description="the true label of the input",
            ),
            "predicted_label": feature.Value(
                dtype=feature.DataType.STRING,
                description="the predicted label",
            ),
            "confidence": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the confidence of the predicted label",
                max_value=1.0,
                min_value=0.0,
                optional=True,
            ),
            "text_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="text length in tokens",
                func=lambda info, x, c: count_tokens(info, x["text"], side="source"),
            ),
            "text_chars": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="text length in characters",
                func=lambda info, x, c: len(x["text"]),
            ),
            "entity_number": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="number of named entities in the text",
                func=lambda info, x, c: len(get_named_entities(x["text"])),
            ),
            "aspect_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="aspect length in tokens",
                func=lambda info, x, c: count_tokens(info, x["aspect"], side="source"),
            ),
            "aspect_position": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="relative position of the aspect in the text",
                func=lambda info, x, c: float(x["text"].find(x["aspect"]))
                / len(x["text"]),
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
        """See Processor.default_analyses."""
        features = self.default_analysis_levels()[0].features
        # Create analyses
        analyses: list[Analysis] = [
            BucketAnalysis(
                level="example",
                description=features["true_label"].description,
                feature="true_label",
                method="discrete",
                num_buckets=15,
            ),
            CalibrationAnalysis(
                level="example",
                description="calibration analysis",
                feature="confidence",
                num_buckets=10,
            ),
            ComboCountAnalysis(
                level="example",
                description="confusion matrix",
                features=("true_label", "predicted_label"),
            ),
        ]
        # Continuous features
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):
        return {}

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {"Accuracy": AccuracyConfig()}
