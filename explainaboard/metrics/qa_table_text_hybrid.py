from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.metrics.registry import register_metric_config
from explainaboard.utils.eval_utils import qa_table_text_hybrid as eval_util
from explainaboard.utils.preprocessor import ExtractiveQAPreprocessor, Preprocessor
from explainaboard.utils.typing_utils import unwrap_or


class HybridQAMetric(Metric):
    """
    An abstract class for HybridQA tasks that measures scores after normalization.
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

            stat_values = []
            for prediction_string in prediction_strings:
                for ground_truth_answer_string in ground_truth_answer_strings:
                    stat_values.append(
                        self.sample_level_metric(
                            prediction_string, ground_truth_answer_string, preprocessor
                        )
                    )

            stat_list.append(max(stat_values))
        return SimpleMetricStats(np.array(stat_list))

    @abc.abstractmethod
    def sample_level_metric(
        self, ground_truth: list, prediction: list, preprocessor: Preprocessor
    ) -> float:
        """
        Calculate a score given a ground truth answer string and a prediction.
        """
        ...


@dataclass
@register_metric_config
class ExactMatchHybridQAConfig(MetricConfig):
    def to_metric(self):
        return ExactMatchHybridQA(self)


class ExactMatchHybridQA(HybridQAMetric):
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
        self, ground_truth: list, prediction: list, preprocessor: Preprocessor
    ) -> float:
        ground_truth = eval_util._answer_to_bags(ground_truth)
        prediction = eval_util._answer_to_bags(prediction)

        exact_match = 0.0
        if set(prediction[0]) == set(ground_truth[0]) and len(prediction[0]) == len(
            ground_truth[0]
        ):
            exact_match = 1.0
        return exact_match


@dataclass
@register_metric_config
class F1ScoreHybridQAConfig(MetricConfig):
    def to_metric(self):
        return F1ScoreHybridQA(self)


class F1ScoreHybridQA(HybridQAMetric):
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
        self, ground_truth: list, prediction: list, preprocessor: Preprocessor
    ):
        ground_truth = eval_util._answer_to_bags(ground_truth)
        prediction = eval_util._answer_to_bags(prediction)
        f1_per_bag = eval_util._align_bags(prediction[1], ground_truth[1])
        f1 = np.mean(f1_per_bag)
        f1 = round(f1, 2)

        return f1
