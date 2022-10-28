"""A processor for the text pair classification task."""

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
from explainaboard.analysis.feature_funcs import (
    accumulate_vocab_from_samples,
    count_tokens,
    feat_freq_rank,
    feat_num_oov,
    get_similarity_by_sacrebleu,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.utils.typing_utils import unwrap


class TextPairClassificationProcessor(Processor):
    """A processor for the text pair classification task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.text_pair_classification

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features: dict[str, FeatureType] = {
            "text1": feature.Value(
                dtype=feature.DataType.STRING,
                description="the first text",
            ),
            "text2": feature.Value(
                dtype=feature.DataType.STRING,
                description="the second text",
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
            "text1_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="text1 length in tokens",
                func=lambda info, x, c: count_tokens(info, x["text1"], side="source"),
            ),
            "text2_length": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="text2 length in tokens",
                func=lambda info, x, c: count_tokens(info, x["text2"], side="target"),
            ),
            "similarity": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the two texts' similarity",
                func=lambda info, x, c: get_similarity_by_sacrebleu(
                    x["text1"], x["text2"]
                ),
            ),
            "text1_divided_text2": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="ratio of two texts' lengths",
                func=lambda info, x, c: c.features["text1_length"]
                / c.features["text2_length"],
            ),
            "num_oov": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of out-of-vocabulary words",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x["text1"], stat["source_vocab"], side="source"
                )
                + feat_num_oov(info, x["text2"], stat["target_vocab"], side="target"),
            ),
            "fre_rank": feature.Value(
                dtype=feature.DataType.FLOAT,
                description=(
                    "the average rank of each word based on its frequency in "
                    "training set"
                ),
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x["text1"], stat["source_vocab_rank"], side="source"
                )
                + feat_freq_rank(
                    info, x["text2"], stat["target_vocab_rank"], side="target"
                ),
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
        analyses.extend(self.continuous_feature_analyses())
        return analyses

    @classmethod
    def default_metrics(
        cls,
        level: str = "example",
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> dict[str, MetricConfig]:
        """See Processor.default_metrics."""
        return {"Accuracy": AccuracyConfig()}

    def _statistics_func(self, samples: Iterable[Any], sys_info: SysOutputInfo):

        samples_list = list(samples)
        source_vocab, source_vocab_rank = accumulate_vocab_from_samples(
            samples_list,
            lambda x: x["text1"],
            unwrap(sys_info.source_tokenizer),
        )

        target_vocab, target_vocab_rank = accumulate_vocab_from_samples(
            samples_list,
            lambda x: x["text2"],
            unwrap(sys_info.target_tokenizer),
        )

        return {
            "source_vocab": source_vocab,
            "source_vocab_rank": source_vocab_rank,
            "target_vocab": target_vocab,
            "target_vocab_rank": target_vocab_rank,
        }
