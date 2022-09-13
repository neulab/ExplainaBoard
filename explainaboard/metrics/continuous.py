"""Evaluation metrics for continuous prediction tasks such as regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.metrics.registry import metric_config_registry


@dataclass
@metric_config_registry.register("RootMeanSquaredErrorConfig")
class RootMeanSquaredErrorConfig(MetricConfig):
    """Configuration for RootMeanSquaredError."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return RootMeanSquaredError(self)


class RootMeanSquaredError(Metric):
    """Calculate the root mean squared error of continuous values."""

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        error = np.array([(x - y) for x, y in zip(true_data, pred_data)])
        squared_error = error * error
        return SimpleMetricStats(squared_error)

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return False

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        return np.sqrt(agg_stats)


@dataclass
@metric_config_registry.register("AbsoluteErrorConfig")
class AbsoluteErrorConfig(MetricConfig):
    """Configuration for AbsoluteError."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return AbsoluteError(self)


class AbsoluteError(Metric):
    """Calculate the absolute error of continuous values."""

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        error = np.array([abs(x - y) for x, y in zip(true_data, pred_data)])
        return SimpleMetricStats(error)
