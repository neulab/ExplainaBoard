from __future__ import annotations

import abc
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.metrics.registry import metric_config_registry
from explainaboard.utils.preprocessor import ExtractiveQAPreprocessor, Preprocessor
from explainaboard.utils.typing_utils import unwrap_or


class ExtractiveQAMetric(Metric):
    """
    An abstract class for extractive QA tasks that measures scores after normalization.
    The actual metric must inherit this class and implement the sample_level_metric()
    function.
    """

    def calc_stats_from_data(
        self,
        true_data: list[Union[str, list[str]]],
        pred_data: list[str],
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:
        true_data = [[x] if isinstance(x, str) else x for x in true_data]
        config = unwrap_or(config, self.config)
        preprocessor = ExtractiveQAPreprocessor(language=config.source_language)
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
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ) -> float:
        """
        Calculate a score given a ground truth answer string and a prediction.
        """
        ...


@dataclass
@metric_config_registry.register("ExactMatchQAConfig")
class ExactMatchQAConfig(MetricConfig):
    def to_metric(self):
        return ExactMatchQA(self)


class ExactMatchQA(ExtractiveQAMetric):
    """
    Calculate a score for extractive QA based on exact match.
    """

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ) -> float:
        return 1.0 if preprocessor(prediction) == preprocessor(ground_truth) else 0.0


@dataclass
@metric_config_registry.register("F1ScoreQAConfig")
class F1ScoreQAConfig(MetricConfig):
    def to_metric(self):
        return F1ScoreQA(self)


class F1ScoreQA(ExtractiveQAMetric):
    """
    Calculate a score for extractive QA based on F1 score.
    """

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ):
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
