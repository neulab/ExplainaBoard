from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from eaas.async_client import AsyncRequest
import numpy as np
import sacrebleu

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.utils.typing_utils import unwrap


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
            self._data = np.array(result['scores'][self.pos]['stats'])

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


@dataclass
class EaaSMetricConfig(MetricConfig):
    def to_metric(self):
        return EaaSMetric(self)


class EaaSMetric(Metric):
    """
    A metric that calculates evaluation scores using EaaS.
    """

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        if self.config.name == 'bleu':
            bleu_class = sacrebleu.BLEU()
            return bleu_class._compute_score_from_stats(list(agg_stats)).score / 100.0
        elif self.config.name == 'chrf':
            chrf_class = sacrebleu.CHRF()
            return chrf_class._compute_score_from_stats(list(agg_stats)).score / 100.0
        elif self.config.name == 'length_ratio':
            return float(agg_stats[0]) / agg_stats[1]
        elif self.config.name == 'length':
            return float(agg_stats[0])
        else:
            return float(agg_stats)

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
        raise NotImplementedError
