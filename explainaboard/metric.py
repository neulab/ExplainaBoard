from __future__ import annotations

import abc
import itertools
from typing import Any, Optional, Union

import numpy as np


class MetricResult:
    def __init__(
        self,
        name: str,
        value: float,
        conf_interval: Optional[tuple[float, float]] = None,
        conf_value: Optional[float] = None,
    ):
        """Initialize a result of an evaluation metric

        :param name: name of the metric
        :param value: value of the metric
        :param conf_interval: the confidence interval of the metric
        :param conv_value: the p-value of the confidence interval
        """
        self.name = name
        self.value = value
        self.conf_interval = conf_interval
        self.conf_value = conf_value

    def to_dict(self):
        ret = {
            'name': self.name,
            'value': self.value,
        }
        if self.conf_interval:
            ret['conf_interval'] = self.conf_interval
        if self.conf_value:
            ret['conf_value'] = self.conf_value
        return ret


class MetricStats:
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def filter(self, indices: Union[list[int], np.ndarray]):
        """
        Return a view of these stats filtered down to the indicated indices
        """
        sdata = self.data
        if type(indices) != np.ndarray:
            indices = np.array(indices)
        return MetricStats(sdata[indices])


class Metric:
    @classmethod
    @abc.abstractmethod
    def default_name(cls) -> str:
        """Returns the default name of the metric."""
        ...

    def __init__(self, name: str = None):
        self.name = name if name else self.default_name()

    @abc.abstractmethod
    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """From a list of true data and predicted data, calculate the sufficient statistics for each data example so
        that the evaluation metric can be calculated later. In the simplest form, this is just the evaluation metric
        value for each example.
        :param true_data: gold-standard data
        :param pred_data: predicted data
        :return: a numpy array of shape [len(true_data), X] where X=1 in the simplest case of decomposable eval metrics
        """
        ...

    def aggregate_stats(self, stats: MetricStats) -> MetricStats:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """
        return MetricStats(np.mean(stats.data, axis=0))

    def calc_metric_from_stats(self, stats: MetricStats) -> float:
        """From aggregated sufficient statistics, calculate the metric value
        :param stats: aggregated statistics
        :return: a single scalar metric value
        """
        if stats.data.size == 1:
            return float(stats.data)
        else:
            raise NotImplementedError

    def bootstrap_interval(
        self,
        stats: MetricStats,
        conf_value: float,
        n_samples: int = 2000,
        prop_samples: float = 0.5,
    ) -> tuple[float, float]:
        """
        :param stats: sufficient statistics as calculated by calc_stats_from_data
        :param conf_value: the p-value of the interval
        :param n_times: the number of bootstrapping samples
        :param prop_samples: the proportion of samples to sample each time
        """
        if conf_value <= 0.0 or conf_value >= 1.0:
            raise ValueError(f'Bad confidence value {conf_value}')
        n_elems = int(prop_samples * len(stats))
        samp_results = np.zeros(shape=(n_samples,))
        all_indices = np.array(range(len(stats)))
        for i in range(n_samples):
            indices = np.random.choice(all_indices, size=n_elems, replace=True)
            samp_stats = self.aggregate_stats(stats.filter(indices))
            samp_results[i] = self.calc_metric_from_stats(samp_stats)
        np.sort(samp_results)
        low = int(n_samples * conf_value / 2.0)
        high = int(n_samples * (1.0 - conf_value / 2.0))
        return samp_results[low], samp_results[high]

    def evaluate_from_stats(
        self, stats: MetricStats, conf_value: Optional[float] = None
    ):
        """Return an evaluation result over stats.
        :param stats: pre-computed metric stats
        :param indices: optionally, the indices to be included in the calculation
        :param conf_value: if set to not None, must be a number between 0 and 1, indicating the p-value of confidence
        intervals
        :return: a resulting metric value
        """
        stats = self.aggregate_stats(stats)
        value = self.calc_metric_from_stats(stats)
        conf_interval = (
            self.bootstrap_interval(stats, conf_value) if conf_value else None
        )
        return MetricResult(self.name, value, conf_interval, conf_value)

    def evaluate(
        self, true_data: list, pred_data: list, conf_value: Optional[float] = None
    ) -> MetricResult:
        """Return an evaluation result over true data and predicted data.
        :param true_data: gold-standard data
        :param pred_data: predicted data
        :param conf_value: if set to not None, must be a number between 0 and 1, indicating the p-value of confidence
        intervals
        :return: a resulting metric value
        """
        stats = self.calc_stats_from_data(true_data, pred_data)
        return self.evaluate_from_stats(stats, conf_value)

    def get_metadata(self) -> dict:
        """Return metadata describing the metric in a reproducible way"""
        return {'name': self.name}


class Accuracy(Metric):
    def default_name(cls) -> str:
        return 'Accuracy'

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> np.ndarray:
        return np.array(
            [(1.0 if x == y else 0.0) for x, y in zip(true_data, pred_data)]
        )


class F1Score(Metric):
    def default_name(cls) -> str:
        return 'F1'

    def __init__(self, average: str = 'micro'):
        self.average = average
        supported_averages = ['micro', 'macro']
        if average not in supported_averages:
            raise ValueError(f'only {supported_averages} supported for now')
        super().__init__(name=self.default_name() + average)

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> np.ndarray:
        id_map = {}
        for word in itertools.chain(true_data, pred_data):
            if word not in id_map:
                id_map[word] = len(id_map)
        n_data = len(true_data)
        n_classes = 3 * len(id_map)
        # This is a bit memory inefficient if there's a large number of classes
        stats = np.zeros((n_data, n_classes * 3))
        for i, (t, p) in enumerate(zip(true_data, pred_data)):
            stats[i, id_map[t] * 3] += 1
            stats[i, id_map[p] * 3 + 1] += 1
            if t == p:
                stats[i, id_map[t] * 3 + 2] += 1
        return stats

    def calc_metric_from_stats(self, stats: np.ndarray) -> float:
        assert len(stats) % 3 == 0
        n_classes = int(len(stats) / 3)
        if self.average == 'micro':
            match, true, pred = 0.0, 0.0, 0.0
            for i in range(n_classes):
                true += stats[i * 3]
                pred += stats[i * 3 + 1]
                match += stats[i * 3 + 1]
            p = match / pred if pred else 0.0
            r = match / true if true else 0.0
            f1_total = 2 * p * r / (p + r)
        elif self.average == 'macro':
            f1_total = 0.0
            for i in range(n_classes):
                true, pred, match = stats[i * 3 : i * 3 + 3]
                p = match / pred if pred else 0.0
                r = match / true if true else 0.0
                f1 = 2 * p * r / (p + r)
                f1_total += f1
            f1_total /= n_classes
        else:
            raise NotImplementedError
        return f1_total

    def get_metadata(self) -> dict:
        meta = dict(super().get_metadata())
        meta['average'] = self.average
        return meta


class Hits(Metric):
    def default_name(cls) -> str:
        return 'Hits'

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> np.ndarray:
        return np.array(
            [(1.0 if t in p else 0.0) for t, p in zip(true_data, pred_data)]
        )


class MeanReciprocalRank(Metric):
    def default_name(cls) -> str:
        return 'MRR'

    def mrr_val(self, true: Any, preds: list):
        true_rank = list(preds).index(true) + 1  # 1-indexed
        return 1.0 / true_rank

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> np.ndarray:
        return np.array([self.mrr_val(t, p) for t, p in zip(true_data, pred_data)])
