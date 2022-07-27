from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Optional

import numpy as np

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config

HUMAN_METRICS = ["LikertScore_fluency", "LikertScore_coherence"]


@dataclass
@register_metric_config
class LikertScoreConfig(MetricConfig):
    aspect: str = "fluency"
    n_annotators: int = 3
    agreement: float = 0.0

    def to_metric(self):
        return LikertScore(self)


class LikertScore(Metric):
    """
    Calculates the hits metric, telling whether the predicted output is in a set of true
    outputs.
    """

    def is_simple_average(self, stats: MetricStats):
        return False

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        config = cast(LikertScoreConfig, self._get_config(config))

        # TODO(Calculate Agreement)

        # "-1" indicates samples to be evaluated
        return MetricStats(
            np.array(
                [[-1.0] * config.n_annotators for t, p in zip(true_data, pred_data)]
            )
        )

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """
        data = stats.get_data()
        if data.size == 0:
            return np.array(0.0)
        else:
            return np.mean(np.mean(data, axis=-1), axis=-1)  # this could be redefined
