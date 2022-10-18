"""Evaluation metrics for extractive question answering."""

from __future__ import annotations

import abc
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.serialization import common_registry
from explainaboard.utils.preprocessor import ExtractiveQAPreprocessor


class ExtractiveQAMetric(Metric):
    """Abstract class for extractive QA tasks that measures scores after normalization.

    The actual metric must inherit this class and implement the sample_level_metric()
    function.
    """

    def calc_stats_from_data(
        self,
        true_data: list[Union[str, list[str]]],
        pred_data: list[str],
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        true_data = [[x] if isinstance(x, str) else x for x in true_data]
        preprocessor = ExtractiveQAPreprocessor(language=self.config.source_language)
        return SimpleMetricStats(
            np.array(
                [
                    max([self.sample_level_metric(t, p, preprocessor) for t in ts])
                    for ts, p in zip(true_data, pred_data)
                ]
            )
        )

    @abc.abstractmethod
    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Callable[[str], str]
    ) -> float:
        """Calculate the metric  for a single sample.

        Args:
            ground_truth: The ground truth answer.
            prediction: The prediction.
            preprocessor: The preprocessor to be applied to the prediction.

        Returns:
            The value of the sample-level metric.
        """
        ...


@dataclass
@common_registry.register("ExactMatchQAConfig")
class ExactMatchQAConfig(MetricConfig):
    """Configuration for ExactMatchQA."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return ExactMatchQA(self)


class ExactMatchQA(ExtractiveQAMetric):
    """Calculate a score for extractive QA based on exact match."""

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Callable[[str], str]
    ) -> float:
        """See ExtractiveQAMetric.sample_level_metric."""
        return 1.0 if preprocessor(prediction) == preprocessor(ground_truth) else 0.0


@dataclass
@common_registry.register("F1ScoreQAConfig")
class F1ScoreQAConfig(MetricConfig):
    """Configuration for F1ScoreQA."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return F1ScoreQA(self)


class F1ScoreQA(ExtractiveQAMetric):
    """Calculate a score for extractive QA based on F1 score."""

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Callable[[str], str]
    ):
        """See ExtractiveQAMetric.sample_level_metric."""
        prediction_tokens = preprocessor(prediction).split()
        ground_truth_tokens = preprocessor(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
