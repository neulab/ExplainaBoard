from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Union

from eaas.async_client import AsyncClient, AsyncRequest
from eaas.config import Config
import numpy as np
import sacrebleu

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.utils.typing_utils import unwrap

_eaas_config = None
_eaas_client = None


def get_eaas_client():
    global _eaas_config, _eaas_client
    if not _eaas_client:
        _eaas_config = Config()
        _eaas_client = AsyncClient(_eaas_config)
    return _eaas_client


class EaaSMetricStats(MetricStats):
    """
    Stats from EaaS for calculation of any of the metrics. These are calculated lazily,
    so that a request is dispatched to the EaaS server and the results are retrieved
    when they're needed.
    """

    def __init__(self, name: str, pos: int, eaas_request: AsyncRequest):
        super().__init__(data=None)
        self.name = name
        self.pos = pos
        self.eaas_request = eaas_request
        self._data: Optional[np.ndarray] = None

        # TODO(odashi): remove this field: this is private but unused.
        self._corpus_value = None

    def __len__(self):
        return len(self.get_data())

    def _fetch_results(self):
        if self._data is None:
            result = self.eaas_request.get_result()
            self._corpus_value = result['scores'][self.pos]['corpus']
            self._data = np.array(
                [
                    x if isinstance(x, list) else [x]
                    for x in result['scores'][self.pos]['stats']
                ]
            )

    def get_corpus_value(self) -> float:
        """
        Return the evaluation metric value over all examples in the corpus.
        """
        self._fetch_results()
        return unwrap(self._corpus_value)

    def get_data(self) -> np.ndarray:
        self._fetch_results()
        return unwrap(self._data)

    def filter(self, indices: Union[list[int], np.ndarray]) -> MetricStats:
        """
        Return a view of these stats filtered down to the indicated indices
        """
        sdata: np.ndarray = self.get_data()
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        return MetricStats(sdata[indices])


# NOTE(odashi): Not register this config to the registry.
# This metric class has different usage than other metrics.
@dataclass
class EaaSMetricConfig(MetricConfig):
    def to_metric(self):
        return EaaSMetric(self)


class EaaSMetric(Metric):
    """
    A metric that calculates evaluation scores using EaaS.
    """

    _NOT_SIMPLE_METRICS = {'bleu', 'chrf', 'length_ratio', 'length'}

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> np.ndarray:
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
        return self.config.name not in self._NOT_SIMPLE_METRICS

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """
        if self.config.name in {'bleu', 'chrf'}:
            return np.sum(stats.get_data(), axis=-2)
        else:
            return np.mean(stats.get_data(), axis=-2)

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        # Note that it's better to batch requests when possible, e.g. as in
        # `processors/conditional_generation.py`
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
