from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Optional

import numpy as np

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats


@dataclass
class LogProbConfig(MetricConfig):
    # If false, return log probability, if true return perplexity
    ppl: bool = False

    def to_metric(self):
        return LogProb(self)


class LogProb(Metric):
    """
    Calculate the log probability
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """
        Take in a list of floats (token-level), or list of lists of floats (sentence
        level) and either one float for each or float+length rows
        """
        if len(pred_data) == 0 or isinstance(pred_data[0], float):
            return MetricStats(np.array(pred_data))
        elif isinstance(pred_data[0], list):
            return MetricStats(np.array([[sum(x), len(x)] for x in pred_data]))
        else:
            t = type(pred_data[0])
            raise ValueError(f'Invalid type of pred_data for calc_stats_from_data {t}')

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        """From aggregated sufficient statistics, calculate the metric value
        :param agg_stats: aggregated statistics
        :param config: a configuration to over-ride the default for this object
        :return: a single scalar metric value
        """
        config = cast(LogProbConfig, self._get_config(config))
        val = (
            float(agg_stats)
            if (isinstance(agg_stats, float) or agg_stats.size == 1)
            else agg_stats[0] / agg_stats[1]
        )
        if config.ppl:
            val = np.exp(-val)
        return val
