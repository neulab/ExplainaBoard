from __future__ import annotations

import abc
from dataclasses import dataclass
import itertools
from typing import Optional

import numpy as np

from explainaboard.metrics.auxiliary import qa_table_text_hybrid_auxiliary as eval_util
from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.metrics.registry import metric_config_registry
from explainaboard.utils.preprocessor import ExtractiveQAPreprocessor, Preprocessor
from explainaboard.utils.typing_utils import unwrap_or


class QATatMetric(Metric):
    """
    An abstract class for HybridQA tasks (see more details about this task:
    https://nextplusplus.github.io/TAT-QA/ ) that measures scores after normalization.
    The actual metric must inherit this class and implement the sample_level_metric()
    function.
    """

    def is_simple_average(self, stats: MetricStats):
        """
        Whether the evaluation score is a simple average of the sufficient statistics.
        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return True

    def calc_stats_from_data(
        self,
        true_data: list,
        pred_data: list,
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:

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

            config = unwrap_or(config, self.config)
            preprocessor = ExtractiveQAPreprocessor(language=config.source_language)

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
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ) -> float:
        """
        Calculate a score given a ground truth answer string and a prediction.
        """
        ...


@dataclass
@metric_config_registry.register("ExactMatchQATatConfig")
class ExactMatchQATatConfig(MetricConfig):
    def to_metric(self):
        return ExactMatchQATat(self)


class ExactMatchQATat(QATatMetric):
    """
    Calculate a score for extractive QA based on exact match.
    """

    def is_simple_average(self, stats: MetricStats):
        """
        Whether the evaluation score is a simple average of the sufficient statistics.
        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return True

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ) -> float:
        ground_truths = eval_util._answer_to_bags(ground_truth)
        predictions = eval_util._answer_to_bags(prediction)

        return float(sorted(predictions[0]) == sorted(ground_truths[0]))


@dataclass
@metric_config_registry.register("F1ScoreQATatConfig")
class F1ScoreQATatConfig(MetricConfig):
    def to_metric(self):
        return F1ScoreQATat(self)


class F1ScoreQATat(QATatMetric):
    """
    Calculate a score for extractive QA based on F1 score.
    """

    def is_simple_average(self, stats: MetricStats):
        """
        Whether the evaluation score is a simple average of the sufficient statistics.
        If so the t-test is applicable, which is much more efficient. Otherwise we do
        bootstrapping to calculate confidence interval, which is slower and potentially
        less effective.
        """
        return True

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ):
        ground_truths = eval_util._answer_to_bags(ground_truth)
        predictions = eval_util._answer_to_bags(prediction)
        f1_per_bag = eval_util._align_bags(predictions[1], ground_truths[1])
        f1 = np.mean(f1_per_bag)

        return f1
