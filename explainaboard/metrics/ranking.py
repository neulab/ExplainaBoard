from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Optional

import numpy as np

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config


@dataclass
@register_metric_config
class HitsConfig(MetricConfig):
    hits_k: int = 5

    def to_metric(self):
        return Hits(self)


class Hits(Metric):
    """
    Calculates the hits metric, telling whether the predicted output is in a set of true
    outputs.
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:  # TODO(Pengfei): why do we need the 3rd argument?
        config = cast(HitsConfig, self._get_config(config))
        return MetricStats(
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
        config = cast(HitsConfig, self._get_config(config))
        return MetricStats(
            np.array([(1.0 if rank <= config.hits_k else 0.0) for rank in rank_data])
        )


@dataclass
@register_metric_config
class MeanReciprocalRankConfig(MetricConfig):
    def to_metric(self):
        return MeanReciprocalRank(self)


class MeanReciprocalRank(Metric):
    """
    Calculates the mean reciprocal rank, 1/rank(true_output) where rank(true_output) is
    the rank of the true output in the predicted n-best list.
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
        return MetricStats(
            np.array([self.mrr_val(t, p) for t, p in zip(true_data, pred_data)])
        )

    def calc_stats_from_rank(
        self, rank_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        if any([rank is None for rank in rank_data]):
            raise ValueError('cannot calculate statistics when rank is none')
        return MetricStats(np.array([1.0 / rank for rank in rank_data]))


@dataclass
@register_metric_config
class MeanRankConfig(MetricConfig):
    def to_metric(self):
        return MeanRank(self)


class MeanRank(Metric):
    """
    Calculates the mean rank, rank(true_output), the rank of the true output in the
    predicted n-best list.
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
        return MetricStats(
            np.array([self.mr_val(t, p) for t, p in zip(true_data, pred_data)])
        )

    def calc_stats_from_rank(
        self, rank_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        return MetricStats(np.array([rank for rank in rank_data if rank is not None]))
