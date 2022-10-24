"""Evaluation metrics to measure log probabilities for language modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.serialization import common_registry
from explainaboard.utils.typing_utils import narrow


@dataclass
@common_registry.register("LogProbConfig")
class LogProbConfig(MetricConfig):
    """Configuration for LogProb metrics.

    Args:
        ppl: Whether to exponente the log prob into perplexity.
    """

    ppl: bool = False

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return LogProb(self)


class LogProb(Metric):
    """Calculate the log probability or perplexity."""

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return stats.num_statistics() == 1

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """See Metric.calc_stats_from_data.

        Takes in a list of floats (token-level), or list of lists of floats (sentence
        level) and either one float for each or float+length rows
        """
        if len(pred_data) == 0 or isinstance(pred_data[0], float):
            return SimpleMetricStats(np.array(pred_data))
        elif isinstance(pred_data[0], list):
            return SimpleMetricStats(np.array([[sum(x), len(x)] for x in pred_data]))
        else:
            t = type(pred_data[0])
            raise ValueError(f"Invalid type of pred_data for calc_stats_from_data {t}")

    def _calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        is_batched = agg_stats.ndim != 1
        if not is_batched:
            agg_stats = agg_stats.reshape((1, agg_stats.shape[0]))
        config = narrow(LogProbConfig, self.config)
        val = (
            agg_stats[:, 0]
            if agg_stats.size == 1
            else agg_stats[:, 0] / agg_stats[:, 1]
        )
        if config.ppl:
            val = np.exp(-val)
        if not is_batched:
            val = val[0]
        return val
