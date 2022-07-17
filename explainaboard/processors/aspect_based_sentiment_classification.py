from __future__ import annotations

from typing import cast, List

from explainaboard import TaskType
from explainaboard.analysis import feature
from explainaboard.analysis.analyses import (
    Analysis,
    AnalysisLevel,
    BucketAnalysis,
    ComboCountAnalysis,
)
from explainaboard.analysis.feature import FeatureType
from explainaboard.analysis.feature_funcs import count_tokens
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.processors.processor_registry import register_processor
from explainaboard.utils.spacy_loader import get_named_entities
from explainaboard.utils.typing_utils import unwrap


@register_processor(TaskType.aspect_based_sentiment_classification)
class AspectBasedSentimentClassificationProcessor(Processor):
    @classmethod
    def task_type(cls) -> TaskType:
        return TaskType.aspect_based_sentiment_classification

    def default_analyses(self) -> list[AnalysisLevel]:
        features: dict[str, FeatureType] = {
            "aspect": feature.Value(
                dtype="string",
                description="the aspect to analyze",
            ),
            "text": feature.Value(
                dtype="string",
                description="the text regarding the aspect",
            ),
            "true_label": feature.Value(
                dtype="string",
                description="the true label of the input",
            ),
            "predicted_label": feature.Value(
                dtype="string",
                description="the predicted label",
            ),
            "text_length": feature.Value(
                dtype="float",
                description="text length in tokens",
                func=lambda info, x, c: count_tokens(info, x['text'], side='source'),
            ),
            "text_chars": feature.Value(
                dtype="float",
                description="text length in characters",
                func=lambda info, x, c: len(x['text']),
            ),
            "entity_number": feature.Value(
                dtype="float",
                description="number of named entities in the text",
                func=lambda info, x, c: len(get_named_entities(x['text'])),
            ),
            "aspect_length": feature.Value(
                dtype="float",
                description="aspect length in tokens",
                func=lambda info, x, c: count_tokens(info, x['aspect'], side='source'),
            ),
            "aspect_position": feature.Value(
                dtype="float",
                description="relative position of the aspect in the text",
                func=lambda info, x, c: float(x["text"].find(x["aspect"]))
                / len(x["text"]),
            ),
        }

        continuous_features = [
            k for k, v in features.items() if ('float' in unwrap(v.dtype))
        ]
        # Create analyses
        analyses: list[Analysis] = [
            BucketAnalysis(
                feature="true_label",
                method="discrete",
                number=15,
            ),
            ComboCountAnalysis(features=("true_label", "predicted_label")),
        ]
        for x in continuous_features:
            analyses.append(BucketAnalysis(x, method="continuous"))

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
        return [AccuracyConfig(name="Accuracy")]
