"""Evaluation metrics for hybrid table-text QA."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
import itertools

import numpy as np

from explainaboard.metrics.auxiliary import qa_table_text_hybrid_auxiliary as eval_util
from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.serialization import common_registry
from explainaboard.utils.preprocessor import ExtractiveQAPreprocessor


class QATatMetric(Metric):
    """An abstract class for HybridQA tasks that measures scores after normalization.

    See more details about this task: https://nextplusplus.github.io/TAT-QA/
    The actual metric must inherit this class and implement the sample_level_metric()
    function.
    """

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return True

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        stat_list = []
        for true_answer_info, pred_answer_info in zip(true_data, pred_data):

            prediction = pred_answer_info["predicted_answer"]
            prediction = prediction if isinstance(prediction, list) else [prediction]

            ground_truth_answer_strings = eval_util.get_answer_str(
                true_answer_info["true_answer"], true_answer_info["answer_scale"]
            )

            prediction_strings = eval_util.get_answer_str(
                prediction, pred_answer_info["predicted_answer_scale"]
            )
            prediction_strings = eval_util.add_percent_pred(
                prediction_strings,
                pred_answer_info["predicted_answer_scale"],
                prediction,
            )

            preprocessor = ExtractiveQAPreprocessor(
                language=self.config.source_language
            )

            args_iter = itertools.product(
                prediction_strings, ground_truth_answer_strings
            )
            stat_values_iter = (
                self.sample_level_metric(y, t, preprocessor) for y, t in args_iter
            )

            stat_list.append(max(stat_values_iter))
        return SimpleMetricStats(np.array(stat_list))

    @abc.abstractmethod
    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Callable[[str], str]
    ) -> float:
        """Calculate a score given a ground truth answer string and a prediction."""
        ...


@dataclass
@common_registry.register("ExactMatchQATatConfig")
class ExactMatchQATatConfig(MetricConfig):
    """Configuration for ExactMatchQATat."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return ExactMatchQATat(self)


class ExactMatchQATat(QATatMetric):
    """Calculate a score for extractive QA based on exact match."""

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return True

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Callable[[str], str]
    ) -> float:
        """See QATatMetric.sample_level_metric."""
        ground_truths = eval_util._answer_to_bags(ground_truth)
        predictions = eval_util._answer_to_bags(prediction)

        return float(sorted(predictions[0]) == sorted(ground_truths[0]))


@dataclass
@common_registry.register("F1ScoreQATatConfig")
class F1ScoreQATatConfig(MetricConfig):
    """Configuration for F1ScoreQATat."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return F1ScoreQATat(self)


class F1ScoreQATat(QATatMetric):
    """Calculate a score for the TAT-QA dataset."""

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return True

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Callable[[str], str]
    ):
        """See QATatMetric.sample_level_metric."""
        ground_truths = eval_util._answer_to_bags(ground_truth)
        predictions = eval_util._answer_to_bags(prediction)
        f1_per_bag = eval_util._align_bags(predictions[1], ground_truths[1])
        f1 = np.mean(f1_per_bag)

        return f1
