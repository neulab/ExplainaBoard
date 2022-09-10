"""Evaluation metrics for ranking-based problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Optional

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.metrics.registry import metric_config_registry
from explainaboard.utils.typing_utils import unwrap_or


@dataclass
@metric_config_registry.register("HitsConfig")
class HitsConfig(MetricConfig):
    """Configuration for Hits.

    Args:
        hits_k: the number of top-k answers to consider in calculation.
    """

    hits_k: int = 5

    def to_metric(self):
        """See MetricConfig.to_metric."""
        return Hits(self)


class Hits(Metric):
    """Calculates the hits metric.

    The metric calculates whether the predicted output is in a set of true outputs.
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""

        config = cast(HitsConfig, unwrap_or(config, self.config))
        return SimpleMetricStats(
            np.array(
                [
                    (1.0 if t in p[: config.hits_k] else 0.0)
                    for t, p in zip(true_data, pred_data)
                ]
            )
        )

    def calc_stats_from_rank(
        self, rank_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:  # TODO(Pengfei): why do we need the 3rd argument?
        config = cast(HitsConfig, unwrap_or(config, self.config))
        return SimpleMetricStats(
            np.array([(1.0 if rank <= config.hits_k else 0.0) for rank in rank_data])
        )


@dataclass
@metric_config_registry.register("MeanReciprocalRankConfig")
class MeanReciprocalRankConfig(MetricConfig):
    """Configuration for MeanReciprocalRank."""

    def to_metric(self):
        """See MetricConfig.to_metric."""
        return MeanReciprocalRank(self)


class MeanReciprocalRank(Metric):
    """Calculates the mean reciprocal rank, 1/rank(true_output).

    rank(true_output) is the rank of the true output in the predicted n-best list.
    """

    @classmethod
    def default_name(cls) -> str:
        return 'MRR'

    def mrr_val(self, true: Any, preds: list):
        if true not in preds:
            return 0.0
        else:
            true_rank = list(preds).index(true) + 1  # 1-indexed
            return 1.0 / true_rank

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        return SimpleMetricStats(
            np.array([self.mrr_val(t, p) for t, p in zip(true_data, pred_data)])
        )

    def calc_stats_from_rank(
        self, rank_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        if any([rank is None for rank in rank_data]):
            raise ValueError('cannot calculate statistics when rank is none')
        return SimpleMetricStats(np.array([1.0 / rank for rank in rank_data]))


@dataclass
@metric_config_registry.register("MeanRankConfig")
class MeanRankConfig(MetricConfig):
    """Configuration for MeanReciprocalRank."""

    def to_metric(self):
        """See MetricConfig.to_metric."""
        return MeanRank(self)


class MeanRank(Metric):
    """Calculates the mean rank of tru_output.

    The metric represents the rank of the true output in the predicted n-best list.
    """

    def mr_val(self, true: Any, preds: list):
        if true not in preds:
            return -1  # placeholder for "infinity"; when `true` is not in `preds`
        else:
            true_rank = list(preds).index(true) + 1  # 1-indexed
            return true_rank

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        return SimpleMetricStats(
            np.array([self.mr_val(t, p) for t, p in zip(true_data, pred_data)])
        )

    def calc_stats_from_rank(
        self, rank_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        return SimpleMetricStats(
            np.array([rank for rank in rank_data if rank is not None])
        )
