from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config


@dataclass
@register_metric_config
class SquaredErrorConfig(MetricConfig):
    def to_metric(self):
        return SquaredError(self)


class SquaredError(Metric):
    """
    Calculate the squared error
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        error = np.array([(x - y) for x, y in zip(true_data, pred_data)])
        squared_error = error * error
        return MetricStats(squared_error)
