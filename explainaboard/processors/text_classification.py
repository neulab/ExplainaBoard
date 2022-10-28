"""A processor for the text classification task."""

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
    count_tokens,
    feat_freq_rank,
    feat_length_freq,
    feat_num_oov,
    get_basic_words,
    get_lexical_richness,
)
from explainaboard.info import SysOutputInfo
from explainaboard.metrics.accuracy import AccuracyConfig
from explainaboard.metrics.metric import MetricConfig
from explainaboard.processors.processor import Processor
from explainaboard.utils.logging import progress
from explainaboard.utils.typing_utils import unwrap


class TextClassificationProcessor(Processor):
    """A processor for the text classification task."""

    @classmethod
    def task_type(cls) -> TaskType:
        """See Processor.task_type."""
        return TaskType.text_classification

    def default_analysis_levels(self) -> list[AnalysisLevel]:
        """See Processor.default_analysis_levels."""
        features: dict[str, FeatureType] = {
            "text": feature.Value(
                dtype=feature.DataType.STRING,
                description="the text of the example",
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
                func=lambda info, x, c: count_tokens(info, x["text"]),
            ),
            "text_chars": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="text length in characters",
                func=lambda info, x, c: len(x["text"]),
            ),
            "basic_words": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the ratio of basic words",
                func=lambda info, x, c: get_basic_words(x["text"]),
            ),
            "lexical_richness": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="lexical diversity",
                func=lambda info, x, c: get_lexical_richness(x["text"]),
            ),
            "num_oov": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the number of out-of-vocabulary words",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_num_oov(
                    info, x["text"], stat["vocab"]
                ),
            ),
            "fre_rank": feature.Value(
                dtype=feature.DataType.FLOAT,
                description=(
                    "the average rank of each word based on its frequency in "
                    "training set"
                ),
                require_training_set=True,
                func=lambda info, x, c, stat: feat_freq_rank(
                    info, x["text"], stat["vocab_rank"]
                ),
            ),
            "length_fre": feature.Value(
                dtype=feature.DataType.FLOAT,
                description="the frequency of text length in training set",
                require_training_set=True,
                func=lambda info, x, c, stat: feat_length_freq(
                    info, x["text"], stat["length_fre"]
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
        vocab: dict[str, float] = {}
        length_fre: dict[int, float] = {}
        total_samps = 0
        tokenizer = unwrap(sys_info.source_tokenizer)
        for sample in progress(samples):
            text = sample["text"]
            tokens = tokenizer(text)
            length = len(tokens)

            length_fre[length] = length_fre.get(length, 0.0) + 1.0

            # update vocabulary
            for w in tokens:
                vocab[w] = vocab.get(w, 0.0) + 1.0

            total_samps += 1

        # the rank of each word based on its frequency
        sorted_dict = {
            key: rank
            for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)
        }
        vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}

        for k, v in length_fre.items():
            length_fre[k] = v * 1.0 / total_samps

        return {"vocab": vocab, "vocab_rank": vocab_rank, "length_fre": length_fre}
