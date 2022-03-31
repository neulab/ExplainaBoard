from __future__ import annotations

import abc
from collections import Counter
from dataclasses import dataclass
import itertools
import re
import string
import sys
from typing import Any, Optional, Union

import numpy as np

from explainaboard.utils.async_eaas import AsyncEaaSRequest
from explainaboard.utils.typing_utils import unwrap


@dataclass
class MetricResult:
    """
    A result of computing a metric over some data
    """

    # Name of the metric that was computed
    name: str
    # Metric value
    value: float
    # Confidence interval of the metric values
    conf_interval: Optional[tuple[float, float]] = None
    # The p-value of the confidence interval
    conf_value: Optional[float] = None

    def to_dict(self):
        ret = {
            'name': self.name,
            'value': self.value,
        }
        if self.conf_interval is not None:
            ret['conf_interval'] = self.conf_interval
        if self.conf_value is not None:
            ret['conf_value'] = self.conf_value
        return ret


class MetricStats:
    """
    A class holding the sufficient statistics necessary to calculate a metric
    """

    def __init__(self, data: Optional[np.ndarray]):
        """
        :param data: A numpy array of dimensions [x,y], where x in the length of the
            dataset, and y is the size of the sufficient statistics necessary to
            calculate the metric.
        """
        self._data = data

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset
        """
        return len(unwrap(self._data))

    def get_data(self) -> np.ndarray:
        """
        Get the sufficient statistics in ndarray format
        """
        return unwrap(self._data)

    def filter(self, indices: Union[list[int], np.ndarray]) -> MetricStats:
        """
        Return a view of these stats filtered down to the indicated indices.
        :param indices: The indices over which the stats should be calculated
        :return: The filtered stats
        """
        sdata = self.get_data()
        if type(indices) != np.ndarray:
            indices = np.array(indices)
        return MetricStats(sdata[indices])


class Metric:
    """
    A class representing an evaluation metric
    """

    @classmethod
    @abc.abstractmethod
    def default_name(cls) -> str:
        """Returns the default name of the metric."""
        ...

    def __init__(self, name: str = None):
        """
        Initialize the metric
        :param name: the name of the metric for reference later
        """
        self.name = name if name else self.default_name()

    @abc.abstractmethod
    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """From a list of true data and predicted data, calculate the sufficient
        statistics for each data example so that the evaluation metric can be calculated
        later. In the simplest form, this is just the evaluation metric value for each
        example.
        :param true_data: gold-standard data
        :param pred_data: predicted data
        :return: a numpy array of shape [len(true_data), X] where X=1 in the simplest
            case of decomposable eval metrics
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
    ) -> MetricResult:
        """Return an evaluation result over stats.
        :param stats: pre-computed metric stats
        :param conf_value: if set to not None, must be a number between 0 and 1,
            indicating the p-value of confidence intervals
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
        :param conf_value: if set to not None, must be a number between 0 and 1,
            indicating the p-value of confidence intervals
        :return: a resulting metric value
        """
        stats = self.calc_stats_from_data(true_data, pred_data)
        return self.evaluate_from_stats(stats, conf_value)

    def get_metadata(self) -> dict:
        """Return metadata describing the metric in a reproducible way"""
        return {'name': self.name}


class Accuracy(Metric):
    """
    Calculate zero-one accuracy, where score is 1 iff the prediction equals the ground
    truth
    """

    @classmethod
    def default_name(cls) -> str:
        return 'Accuracy'

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        return MetricStats(
            np.array([(1.0 if x == y else 0.0) for x, y in zip(true_data, pred_data)])
        )


class F1Score(Metric):
    """
    Calculate F1 score, micro- or macro-averaged over classes. Should match sklearn's
    implementation.
    """

    @classmethod
    def default_name(cls) -> str:
        return 'F1'

    def __init__(self, average: str = 'micro', separate_match: bool = False):
        """Constructor for f-measure
        :param average: What variety of average to measure
        :param separate_match: Whether to count matches separately for true and pred.
            This is useful in, for example bucketing, when ref and pred are not aligned
        """
        self.average: str = average
        self.separate_match: bool = separate_match
        self._stat_mult: int = 4 if separate_match else 3
        self._pred_match_offfset: int = 3 if separate_match else 2
        supported_averages = {'micro', 'macro'}
        if average not in supported_averages:
            raise ValueError(f'only {supported_averages} supported for now')
        super().__init__(name=self.default_name() + average)

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """
        Return sufficient statistics necessary to compute f-score.
        :param true_data: True outputs
        :param pred_data: Predicted outputs
        :return: Returns stats for each class (integer id c) in the following columns of
            MetricStats
            * c*self._stat_mult + 0: occurrences in the true output
            * c*self._stat_mult + 1: occurrences in the predicted output
            * c*self._stat_mult + 2: number of matches with the true output
            * c*self._stat_mult + 3: number of matches with the predicted output
                (when self.separate_match=True only)
        """
        id_map: dict[str, int] = {}
        for word in itertools.chain(true_data, pred_data):
            if word not in id_map:
                id_map[word] = len(id_map)
        n_data = len(true_data)
        n_classes = len(id_map)
        # This is a bit memory inefficient if there's a large number of classes
        stats = np.zeros((n_data, n_classes * self._stat_mult))
        for i, (t, p) in enumerate(zip(true_data, pred_data)):
            tid, pid = id_map[t], id_map[p]
            stats[i, tid * self._stat_mult + 0] += 1
            stats[i, pid * self._stat_mult + 1] += 1
            if tid == pid:
                stats[i, tid * self._stat_mult + 2] += 1
                if self.separate_match:
                    stats[i, tid * self._stat_mult + 3] += 1
        return MetricStats(stats)

    def calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> float:
        n_classes = int(len(agg_stats) / self._stat_mult)
        if self.average == 'micro':
            true, pred, true_match, pred_match = 0.0, 0.0, 0.0, 0.0
            for i in range(n_classes):
                true += agg_stats[i * self._stat_mult + 0]
                pred += agg_stats[i * self._stat_mult + 1]
                true_match += agg_stats[i * self._stat_mult + 2]
                pred_match += agg_stats[i * self._stat_mult + self._pred_match_offfset]
            p = pred_match / pred if pred else 0.0
            r = true_match / true if true else 0.0
            f1_total = 2 * p * r / (p + r) if p + r != 0.0 else 0.0
        elif self.average == 'macro':
            f1_total = 0.0
            for i in range(n_classes):
                true, pred, true_match = agg_stats[
                    i * self._stat_mult : i * self._stat_mult + 3
                ]
                pred_match = agg_stats[i * self._stat_mult + self._pred_match_offfset]
                p = pred_match / pred if pred else 0.0
                r = true_match / true if true else 0.0
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
    """
    Calculates the hits metric, telling whether the predicted output is in a set of true
    outputs.
    """

    @classmethod
    def default_name(cls) -> str:
        return 'Hits'

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        return MetricStats(
            np.array([(1.0 if t in p else 0.0) for t, p in zip(true_data, pred_data)])
        )


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

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        return MetricStats(
            np.array([self.mrr_val(t, p) for t, p in zip(true_data, pred_data)])
        )


class EaaSMetricStats(MetricStats):
    """
    Stats from EaaS for calculation of any of the metrics. These are calculated lazily,
    so that a request is dispatched to the EaaS server and the results are retrieved
    when they're needed.
    """

    def __init__(self, name: str, eaas_request: AsyncEaaSRequest):
        super().__init__(data=None)
        self.name = name
        self.eaas_request = eaas_request
        self._data: Optional[np.ndarray] = None

        # TODO(odashi): remove this field: this is private but unused.
        self._corpus_value = None

    def __len__(self):
        return len(self.get_data())

    def _fetch_results(self):
        if self._data is None:
            result = self.eaas_request.get_result()
            self._corpus_value = result['corpus_level'][f'corpus_{self.name}']
            samps = result['sample_level']
            self._data = np.array([x[self.name] for x in samps])

    def get_corpus_value(self) -> float:
        """
        Return the evaluation metric value over all examples in the corpus.
        """
        self._fetch_results()
        return unwrap(self._corpus_value)

    def get_data(self) -> np.ndarray:
        self._fetch_results()
        return self._data

    def filter(self, indices: Union[list[int], np.ndarray]) -> MetricStats:
        """
        Return a view of these stats filtered down to the indicated indices
        """
        sdata: np.ndarray = unwrap(self._data)
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        return MetricStats(sdata[indices])


class EaaSMetric(Metric):
    """
    A metric that calculates evaluation scores using EaaS.
    """

    @classmethod
    def default_name(cls) -> str:
        raise NotImplementedError

    def __init__(self, name: str):
        super().__init__(name)
        # !!! Temporary warning
        non_decomposable_metrics = ['bleu', 'chrf']
        if name in non_decomposable_metrics:
            print(
                f'WARNING: corpus-level {name} is currently calculated as the average '
                f'of sentence-level {name}, which is not technically correct. '
                'This is a known issue that we are working on: '
                'https://github.com/neulab/ExplainaBoard/issues/161',
                file=sys.stderr,
            )
        # !!! End temporary warning
        self.name = name

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        raise NotImplementedError


class QAMetric(Metric):
    """
    An abstract class for extractive QA tasks that measures scores after normalization.
    The actual metric must inherit this class and implement the sample_level_metric()
    function.
    """

    def normalize_answer(self, s: str) -> str:
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
        """
        Calculate a score given a ground truth answer string and a prediction.
        """
        ...


class ExactMatchQA(QAMetric):
    """
    Calculate a score for extractive QA based on exact match.
    """

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
    """
    Calculate a score for extractive QA based on F1 score.
    """

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
