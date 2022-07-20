from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config


@dataclass
@register_metric_config
class RootMeanSquaredErrorConfig(MetricConfig):
    def to_metric(self):
        return RootMeanSquaredError(self)


class RootMeanSquaredError(Metric):
    """
    Calculate the squared error
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        error = np.array([(x - y) for x, y in zip(true_data, pred_data)])
        squared_error = error * error
        return MetricStats(squared_error)

    def is_simple_average(self, stats: MetricStats):
        return False

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> np.ndarray:
        return np.sqrt(agg_stats)


@dataclass
@register_metric_config
class AbsoluteErrorConfig(MetricConfig):
    def to_metric(self):
        return AbsoluteError(self)


class AbsoluteError(Metric):
    """
    Calculate the squared error
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        error = np.array([abs(x - y) for x, y in zip(true_data, pred_data)])
        return MetricStats(error)
