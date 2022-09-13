"""Evaluation metrics using the "Evaluation as a Service" library."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast, final, Optional

from eaas.async_client import AsyncClient, AsyncRequest
from eaas.config import Config
import numpy as np
import sacrebleu

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import metric_config_registry

_eaas_config = None
_eaas_client = None


def get_eaas_client():
    """Get a global client for EaaS."""
    global _eaas_config, _eaas_client
    if not _eaas_client:
        _eaas_config = Config()
        _eaas_client = AsyncClient(_eaas_config)
    return _eaas_client


@final
class EaaSMetricStats(MetricStats):
    """MetricStats with EaaS invocations.

    Obtaining the data from EaaS is deferred until it is wanted.
    """

    def __init__(self, name: str, pos: int, eaas_request: AsyncRequest) -> None:
        """Initializes the EaaSMetricStats.

        Args:
            name: Name of this metric.
            pos: Position of the statistics in the returned array.
            eaas_request: Request object to the EaaS service.
        """
        self._name = name  # TODO(odashi): Remove this member.
        self._pos = pos
        self._eaas_request = eaas_request
        self._data: np.ndarray | None = None

    def _fetch_results(self) -> None:
        """Obtains the data from the EaaS service."""
        if self._data is None:
            result = self._eaas_request.get_result()
            self._data = np.array(
                [
                    x if isinstance(x, list) else [x]
                    for x in result['scores'][self._pos]['stats']
                ]
            )

    def __len__(self) -> int:
        """See MetricStats.__len__."""
        return len(self.get_data())

    def is_batched(self) -> bool:
        """See MetricStats.is_batched."""
        return False

    def num_statistics(self) -> int:
        """See MetricStats.num_statistics."""
        return self.get_data().shape[-1]

    def get_data(self) -> np.ndarray[tuple[int, int], Any]:
        """See MetricStats.get_data."""
        self._fetch_results()
        # self._data must have the data at this point.
        return cast(np.ndarray, self._data)

    def get_batch_data(self) -> np.ndarray[tuple[int, int, int], Any]:
        """See MetricStats.get_batch_data."""
        raise NotImplementedError


@dataclass
@metric_config_registry.register("EaaSMetricConfig")
class EaaSMetricConfig(MetricConfig):
    """Configuration for EaaSMetric."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return EaaSMetric(self)


class EaaSMetric(Metric):
    """A metric that calculates evaluation scores using EaaS."""

    _NOT_SIMPLE_METRICS = {'bleu', 'chrf', 'length_ratio', 'length'}

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        if agg_stats.ndim == 1:
            agg_stats = agg_stats.reshape((1, agg_stats.shape[0]))
        n_samples = agg_stats.shape[0]
        if self.config.name in {'bleu', 'chrf'}:
            ret_metric = np.zeros(n_samples)
            metric_class = (
                sacrebleu.BLEU() if self.config.name == 'bleu' else sacrebleu.CHRF()
            )
            for i, single_stat in enumerate(agg_stats):
                ret_metric[i] = (
                    metric_class._compute_score_from_stats(list(single_stat)).score
                    / 100.0
                )
            return ret_metric
        elif self.config.name == 'length_ratio':
            return agg_stats[:, 0] / agg_stats[:, 1]
        elif self.config.name == 'length':
            return agg_stats[:, 0]
        else:
            return agg_stats

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return self.config.name not in self._NOT_SIMPLE_METRICS

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """See: Metric.aggregate_stats."""
        data = stats.get_batch_data() if stats.is_batched() else stats.get_data()
        if self.config.name in {'bleu', 'chrf'}:
            return np.sum(data, axis=-2)
        else:
            return np.mean(data, axis=-2)

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data.

        Note that specifically for EaaSMetric, it's better to batch requests when
        possible, so they can be sent in a single API call. For example, see
        `processors/conditional_generation.py`.
        """
        inputs = []
        for td, pd in zip(true_data, pred_data):
            ntd = copy.deepcopy(td)
            ntd['hypothesis'] = pd
            inputs.append(ntd)
        async_request = get_eaas_client().async_score(
            inputs,
            metrics=[self.config.name],
            calculate=['corpus', 'stats'],
        )
        return EaaSMetricStats(name=self.config.name, pos=0, eaas_request=async_request)
