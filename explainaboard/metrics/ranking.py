"""Evaluation metrics for ranking-based problems."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.serialization import common_registry


class RankingMetric(Metric, metaclass=abc.ABCMeta):
    """A metric for ranking."""

    @abc.abstractmethod
    def calc_stats_from_rank(self, rank_data: list[int]) -> MetricStats:
        """Calculate statistics from rank data.

        Args:
            rank_data: A list of integer ranks.

        Returns:
            The aggregate statistics for this metric.
        """
        ...


@dataclass
@common_registry.register("HitsConfig")
class HitsConfig(MetricConfig):
    """Configuration for Hits.

    Args:
        hits_k: the number of top-k answers to consider in calculation.
    """

    hits_k: int = 5

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return Hits(self)


class Hits(RankingMetric):
    """Calculates the hits metric.

    The metric calculates whether the predicted output is in a set of true outputs.
    """

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        config = cast(HitsConfig, self.config)
        return SimpleMetricStats(
            np.array(
                [
                    (1.0 if t in p[: config.hits_k] else 0.0)
                    for t, p in zip(true_data, pred_data)
                ]
            )
        )

    def calc_stats_from_rank(self, rank_data: list[int]) -> MetricStats:
        """See RankingMetric.calc_stats_from_rank."""
        config = cast(HitsConfig, self.config)
        return SimpleMetricStats(
            np.array([(1.0 if rank <= config.hits_k else 0.0) for rank in rank_data])
        )


@dataclass
@common_registry.register("MeanReciprocalRankConfig")
class MeanReciprocalRankConfig(MetricConfig):
    """Configuration for MeanReciprocalRank."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return MeanReciprocalRank(self)


class MeanReciprocalRank(RankingMetric):
    """Calculates the mean reciprocal rank, 1/rank(true_output).

    rank(true_output) is the rank of the true output in the predicted n-best list.
    """

    def _mrr_val(self, true: Any, preds: list):
        if true not in preds:
            return 0.0
        else:
            true_rank = list(preds).index(true) + 1  # 1-indexed
            return 1.0 / true_rank

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        return SimpleMetricStats(
            np.array([self._mrr_val(t, p) for t, p in zip(true_data, pred_data)])
        )

    def calc_stats_from_rank(self, rank_data: list) -> MetricStats:
        """See RankingMetric.calc_stats_from_rank."""
        if any(rank is None for rank in rank_data):
            raise ValueError("cannot calculate statistics when rank is none")
        return SimpleMetricStats(np.array([1.0 / rank for rank in rank_data]))


@dataclass
@common_registry.register("MeanRankConfig")
class MeanRankConfig(MetricConfig):
    """Configuration for MeanReciprocalRank."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return MeanRank(self)


class MeanRank(RankingMetric):
    """Calculates the mean rank of the true output in a predicted n-best list."""

    def _mr_val(self, true: Any, preds: list):
        if true not in preds:
            return -1  # placeholder for "infinity"; when `true` is not in `preds`
        else:
            true_rank = list(preds).index(true) + 1  # 1-indexed
            return true_rank

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        return SimpleMetricStats(
            np.array([self._mr_val(t, p) for t, p in zip(true_data, pred_data)])
        )

    def calc_stats_from_rank(self, rank_data: list) -> MetricStats:
        """See RankingMetric.calc_stats_from_rank."""
        return SimpleMetricStats(
            np.array([rank for rank in rank_data if rank is not None])
        )
