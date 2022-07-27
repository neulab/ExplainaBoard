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
    aspect:str = "fluency"

    def to_metric(self):
        return LikertScore(self)


class LikertScore(Metric):
    """
    Calculates the hits metric, telling whether the predicted output is in a set of true
    outputs.
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        config = cast(LikertScoreConfig, self._get_config(config))
        return MetricStats(
            np.array(
                [
                    -1.0 for t, p in zip(true_data, pred_data)
                ]
            )
        )
