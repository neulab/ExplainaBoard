from __future__ import annotations

import abc
from collections import Counter
import itertools
import re
import string
import sys
from typing import Any, Optional, Union

import numpy as np

from explainaboard.utils.async_eaas import AsyncEaaSResult


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
        :param conf_value: the p-value of the confidence interval
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
    def __init__(self, data: Optional[np.ndarray]):
        self._data = data

    def __len__(self):
        return len(self._data)

    def get_data(self):
        return self._data

    def filter(self, indices: Union[list[int], np.ndarray]):
        """
        Return a view of these stats filtered down to the indicated indices
        """
        sdata = self.get_data()
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

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """
        return np.mean(stats.get_data(), axis=0)

    def calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> float:
        """From aggregated sufficient statistics, calculate the metric value
        :param agg_stats: aggregated statistics
        :return: a single scalar metric value
        """
        if agg_stats.size == 1:
            return float(agg_stats)
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
        :param n_samples: the number of bootstrapping samples
        :param prop_samples: the proportion of samples to sample each time
        """
        if conf_value <= 0.0 or conf_value >= 1.0:
            raise ValueError(f'Bad confidence value {conf_value}')
        n_elems = int(prop_samples * len(stats))
        samp_results = np.zeros(shape=(n_samples,))
        all_indices = np.array(range(len(stats)))
        for i in range(n_samples):
            indices = np.random.choice(all_indices, size=n_elems, replace=True)
            agg_stats = self.aggregate_stats(stats.filter(indices))
            samp_results[i] = self.calc_metric_from_aggregate(agg_stats)
        samp_results = np.sort(samp_results)
        low = int(n_samples * conf_value / 2.0)
        high = int(n_samples * (1.0 - conf_value / 2.0))
        return samp_results[low], samp_results[high]

    def evaluate_from_stats(
        self, stats: MetricStats, conf_value: Optional[float] = None
    ):
        """Return an evaluation result over stats.
        :param stats: pre-computed metric stats
        :param conf_value: if set to not None, must be a number between 0 and 1, indicating the p-value of confidence
        intervals
        :return: a resulting metric value
        """
        agg_stats = self.aggregate_stats(stats)
        value = self.calc_metric_from_aggregate(agg_stats)
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
    @classmethod
    def default_name(cls) -> str:
        return 'Accuracy'

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        return MetricStats(
            np.array([(1.0 if x == y else 0.0) for x, y in zip(true_data, pred_data)])
        )


class F1Score(Metric):
    @classmethod
    def default_name(cls) -> str:
        return 'F1'

    def __init__(self, average: str = 'micro'):
        self.average = average
        supported_averages = {'micro', 'macro'}
        if average not in supported_averages:
            raise ValueError(f'only {supported_averages} supported for now')
        super().__init__(name=self.default_name() + average)

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        id_map = {}
        for word in itertools.chain(true_data, pred_data):
            if word not in id_map:
                id_map[word] = len(id_map)
        n_data = len(true_data)
        n_classes = len(id_map)
        # This is a bit memory inefficient if there's a large number of classes
        stats = np.zeros((n_data, n_classes * 3))
        for i, (t, p) in enumerate(zip(true_data, pred_data)):
            tid, pid = id_map[t], id_map[p]
            stats[i, tid * 3] += 1
            stats[i, pid * 3 + 1] += 1
            if tid == pid:
                stats[i, tid * 3 + 2] += 1
        return MetricStats(stats)

    def calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> float:
        n_classes = int(len(agg_stats) / 3)
        if self.average == 'micro':
            match, true, pred = 0.0, 0.0, 0.0
            for i in range(n_classes):
                true += agg_stats[i * 3]
                pred += agg_stats[i * 3 + 1]
                match += agg_stats[i * 3 + 2]
            p = match / pred if pred else 0.0
            r = match / true if true else 0.0
            f1_total = 2 * p * r / (p + r) if p + r != 0.0 else 0.0
        elif self.average == 'macro':
            f1_total = 0.0
            for i in range(n_classes):
                true, pred, match = agg_stats[i * 3 : i * 3 + 3]
                p = match / pred if pred else 0.0
                r = match / true if true else 0.0
                f1 = 2 * p * r / (p + r) if p + r != 0.0 else 0.0
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
    @classmethod
    def default_name(cls) -> str:
        return 'Hits'

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        return MetricStats(
            np.array([(1.0 if t in p else 0.0) for t, p in zip(true_data, pred_data)])
        )


class MeanReciprocalRank(Metric):
    @classmethod
    def default_name(cls) -> str:
        return 'MRR'

    def mrr_val(self, true: Any, preds: list):
        if true not in preds:
            return 0.0
        else:
            true_rank = list(preds).index(true) + 1  # 1-indexed
            return 1.0 / true_rank

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        return MetricStats(
            np.array([self.mrr_val(t, p) for t, p in zip(true_data, pred_data)])
        )


class EaaSMetricStats(MetricStats):
    def __init__(self, name: str, eaas_result: AsyncEaaSResult):
        super().__init__(data=None)
        self.name = name
        self.eaas_result = eaas_result
        self._corpus_value = None

    def __len__(self):
        return len(self.get_data())

    def _fetch_results(self):
        if not self._data:
            result = self.eaas_result.get_result()
            self._corpus_value = result['corpus_level'][f'corpus_{self.name}']
            samps = result['sample_level']
            self._data = np.array([x[self.name] for x in samps])

    def get_corpus_value(self) -> float:
        self._fetch_results()
        return self._corpus_value

    def get_data(self) -> np.ndarray:
        self._fetch_results()
        return self._data

    def filter(self, indices: Union[list[int], np.ndarray]):
        """
        Return a view of these stats filtered down to the indicated indices
        """
        sdata = self._data
        if type(indices) != np.ndarray:
            indices = np.array(indices)
        return MetricStats(sdata[indices])


class EaaSMetric(Metric):
    @classmethod
    def default_name(cls) -> str:
        raise NotImplementedError

    def __init__(self, name: str):
        super().__init__(name)
        # !!! Temporary warning
        non_decomposable_metrics = ['bleu']
        if name in non_decomposable_metrics:
            print(
                f'WARNING: corpus-level {name} is currently calculated as the average of sentence-level {name}, which is not technically correct. This is a known issue that we are working on: https://github.com/neulab/ExplainaBoard/issues/161',
                file=sys.stderr,
            )
        # !!! End temporary warning
        self.name = name


class QAMetric(Metric):
    def normalize_answer(self, s: str):
        """Lower text and remove punctuation, articles and extra whitespace."""
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ' '.join(s.split())
        exclude_punc = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude_punc)
        s = s.lower()
        return s

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        return MetricStats(
            np.array(
                [
                    max([self.sample_level_metric(t, p) for t in ts])
                    for ts, p in zip(true_data, pred_data)
                ]
            )
        )

    @abc.abstractmethod
    def sample_level_metric(self, ground_truth: str, prediction: str) -> float:
        ...


class ExactMatchQA(QAMetric):
    @classmethod
    def default_name(cls) -> str:
        return 'ExactMatchQA'

    def sample_level_metric(self, ground_truth: str, prediction: str) -> float:
        return (
            1.0
            if self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
            else 0.0
        )


class F1ScoreQA(QAMetric):
    @classmethod
    def default_name(cls) -> str:
        return 'F1ScoreQA'

    def sample_level_metric(self, ground_truth: str, prediction: str):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
