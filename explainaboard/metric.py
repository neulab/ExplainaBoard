from __future__ import annotations

import abc
from collections import Counter
from dataclasses import dataclass
import itertools
import sys
from typing import Any, cast, Optional, TypeVar, Union

from eaas.async_client import AsyncRequest
from eaas.endpoint import EndpointConfig
import numpy as np
import sacrebleu

from explainaboard.utils.preprocessor import ExtractiveQAPreprocessor, Preprocessor
from explainaboard.utils.span_utils import BIOSpanOps, BMESSpanOps, SpanOps
from explainaboard.utils.typing_utils import unwrap

T = TypeVar('T')


@dataclass
class MetricResult:
    """
    A result of computing a metric over some data
    """

    # Configuration with which it was calculated
    config: MetricConfig
    # Metric value
    value: float
    # Confidence interval of the metric values
    conf_interval: Optional[tuple[float, float]] = None
    # The p-value of the confidence interval
    conf_value: Optional[float] = None

    def to_dict(self):
        ret = {
            'config': self.config.__dict__,
            'value': self.value,
        }
        if self.conf_interval is not None:
            ret['conf_interval'] = self.conf_interval
        if self.conf_value is not None:
            ret['conf_value'] = self.conf_value
        return ret


@dataclass
class MetricConfig(dict):
    """
    The configuration for the metric. This can be passed in to the metric either in
    the constructor (e.g. for compute-intensive operations such as model loading),
    or when performing individual metric computation.
    """

    name: str
    source_language: str | None = None
    target_language: str | None = None
    cls_name: str = ''

    def __post_init__(self):
        # Save the class name
        self.cls_name = type(self).__name__

    def to_metric(self):
        raise NotImplementedError


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

    def __init__(
        self,
        config: MetricConfig,
    ):
        """
        Initialize the metric
        :param config: The configuration for the metric
        """
        self.config: MetricConfig = config

    def _get_config(self, config: Optional[MetricConfig] = None) -> MetricConfig:
        """
        Get the configuration or overwritten configuration
        :param config: Optional configuration to override the default configuration
        :return: Either the default or overridden configuration
        """
        ret_config: MetricConfig = unwrap(config) if config is not None else self.config
        return ret_config

    @abc.abstractmethod
    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """From a list of true data and predicted data, calculate the sufficient
        statistics for each data example so that the evaluation metric can be calculated
        later. In the simplest form, this is just the evaluation metric value for each
        example.
        :param true_data: gold-standard data
        :param pred_data: predicted data
        :param config: a configuration to over-ride the default for this object
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

        data = stats.get_data()
        if data.size == 0:
            return np.array(0.0)
        else:
            return np.mean(data, axis=0)

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        """From aggregated sufficient statistics, calculate the metric value
        :param agg_stats: aggregated statistics
        :param config: a configuration to over-ride the default for this object
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
        n_samples: int = 1000,
        prop_samples: float = 0.5,
        config: Optional[MetricConfig] = None,
    ) -> tuple[float, float]:
        """
        :param stats: sufficient statistics as calculated by calc_stats_from_data
        :param conf_value: the p-value of the interval
        :param n_samples: the number of bootstrapping samples
        :param prop_samples: the proportion of samples to sample each time
        :param config: a configuration to over-ride the default for this object
        """
        if conf_value <= 0.0 or conf_value >= 1.0:
            raise ValueError(f'Bad confidence value {conf_value}')
        n_elems = int(prop_samples * len(stats))
        samp_results = np.zeros(shape=(n_samples,))
        all_indices = np.array(range(len(stats)))
        rng = np.random.default_rng()
        all_indices = rng.choice(all_indices, size=(n_samples, n_elems), replace=True)
        for i in range(n_samples):
            indices = all_indices[i]
            agg_stats = self.aggregate_stats(stats.filter(indices))
            samp_results[i] = self.calc_metric_from_aggregate(agg_stats, config)
        samp_results = np.sort(samp_results)
        low = int(n_samples * conf_value / 2.0)
        high = int(n_samples * (1.0 - conf_value / 2.0))
        return samp_results[low], samp_results[high]

    def evaluate_from_stats(
        self,
        stats: MetricStats,
        conf_value: Optional[float] = None,
        config: Optional[MetricConfig] = None,
    ) -> MetricResult:
        """Return an evaluation result over stats.
        :param stats: pre-computed metric stats
        :param conf_value: if set to not None, must be a number between 0 and 1,
            indicating the p-value of confidence intervals
        :param config: a configuration to over-ride the default for this object
        :return: a resulting metric value
        """
        config = self._get_config(config)
        agg_stats = self.aggregate_stats(stats)
        value = self.calc_metric_from_aggregate(agg_stats, config)
        conf_interval = (
            self.bootstrap_interval(stats, conf_value) if conf_value else None
        )
        return MetricResult(config, value, conf_interval, conf_value)

    def evaluate(
        self,
        true_data: list,
        pred_data: list,
        conf_value: Optional[float] = None,
        config: Optional[MetricConfig] = None,
    ) -> MetricResult:
        """Return an evaluation result over true data and predicted data.
        :param true_data: gold-standard data
        :param pred_data: predicted data
        :param conf_value: if set to not None, must be a number between 0 and 1,
            indicating the p-value of confidence intervals
        :param config: a configuration to over-ride the default for this object
        :return: a resulting metric value
        """
        stats = self.calc_stats_from_data(true_data, pred_data, config)
        return self.evaluate_from_stats(stats, conf_value, config)


@dataclass
class AccuracyConfig(MetricConfig):
    def to_metric(self):
        return Accuracy(self)


class Accuracy(Metric):
    """
    Calculate zero-one accuracy, where score is 1 iff the prediction equals the ground
    truth
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        return MetricStats(
            np.array([(1.0 if x == y else 0.0) for x, y in zip(true_data, pred_data)])
        )


@dataclass
class CorrectCountConfig(MetricConfig):
    def to_metric(self):
        return CorrectCount(self)


class CorrectCount(Accuracy):
    """
    Calculate the absolute value of correct number
    """

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """
        Aggregate sufficient statistics from multiple examples into a single example
        :param stats: stats for every example
        :return: aggregated stats
        """
        data = stats.get_data()
        if data.size == 0:
            return np.array(0.0)
        else:
            return np.sum(data, axis=0)


@dataclass
class F1ScoreConfig(MetricConfig):
    average: str = 'micro'
    separate_match: bool = False
    ignore_classes: Optional[list] = None

    def to_metric(self):
        return F1Score(self)


class F1Score(Metric):
    """
    Calculate F1 score, micro- or macro-averaged over classes. Should match sklearn's
    implementation.
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """
        Return sufficient statistics necessary to compute f-score.
        :param true_data: True outputs
        :param pred_data: Predicted outputs
        :param config: Configuration, if overloading the default for this object
        :return: Returns stats for each class (integer id c) in the following columns of
            MetricStats
            * c*stat_mult + 0: occurrences in the true output
            * c*stat_mult + 1: occurrences in the predicted output
            * c*stat_mult + 2: number of matches with the true output
            * c*stat_mult + 3: number of matches with the predicted output
                (when self.separate_match=True only)
        """
        config = cast(F1ScoreConfig, self._get_config(config))
        stat_mult: int = 4 if config.separate_match else 3

        id_map: dict[str, int] = {}
        if config.ignore_classes is not None:
            for ignore_class in config.ignore_classes:
                id_map[ignore_class] = -1

        for word in itertools.chain(true_data, pred_data):
            if word not in id_map:
                id_map[word] = len(id_map)
        n_data = len(true_data)
        n_classes = len(id_map)
        # This is a bit memory inefficient if there's a large number of classes
        stats = np.zeros((n_data, n_classes * stat_mult))
        for i, (t, p) in enumerate(zip(true_data, pred_data)):
            tid, pid = id_map[t], id_map[p]
            if tid != -1:
                stats[i, tid * stat_mult + 0] += 1
            if pid != -1:
                stats[i, pid * stat_mult + 1] += 1
                if tid == pid:
                    stats[i, tid * stat_mult + 2] += 1
                    if config.separate_match:
                        stats[i, tid * stat_mult + 3] += 1
        return MetricStats(stats)

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:

        if agg_stats.size == 1:
            return float(agg_stats)

        config = cast(F1ScoreConfig, self._get_config(config))
        supported_averages = {'micro', 'macro'}
        stat_mult: int = 4 if config.separate_match else 3
        if config.average not in supported_averages:
            raise ValueError(f'only {supported_averages} supported for now')
        n_classes = int(len(agg_stats) / stat_mult)

        if config.average == 'micro':
            true, pred, true_match, pred_match = 0.0, 0.0, 0.0, 0.0
            for i in range(n_classes):
                true += agg_stats[i * stat_mult + 0]
                pred += agg_stats[i * stat_mult + 1]
                true_match += agg_stats[i * stat_mult + 2]
                pred_match += agg_stats[(i + 1) * stat_mult - 1]
            p = pred_match / pred if pred else 0.0
            r = true_match / true if true else 0.0
            f1_total = 2 * p * r / (p + r) if p + r != 0.0 else 0.0

        elif config.average == 'macro':
            f1_total = 0.0
            for i in range(n_classes):
                true, pred, true_match = agg_stats[i * stat_mult : i * stat_mult + 3]
                pred_match = agg_stats[(i + 1) * stat_mult - 1]
                p = pred_match / pred if pred else 0.0
                r = true_match / true if true else 0.0
                f1 = 2 * p * r / (p + r) if p + r != 0.0 else 0.0
                f1_total += f1
            f1_total /= n_classes
        else:
            raise NotImplementedError
        return f1_total


@dataclass
class SeqCorrectCountConfig(MetricConfig):
    def to_metric(self):
        return SeqCorrectCount(self)


class SeqCorrectCount(CorrectCount):
    def calc_stats_from_data(
        self,
        true_edits_ldl: list[dict[str, list]],
        pred_edits_ldl: list[dict[str, list]],
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:
        def _get_flatten_edits(edits: list[dict]):
            flatten_edits = []
            for edit in edits:
                start_idx, end_idx, corrections = (
                    edit["start_idx"],
                    edit["end_idx"],
                    edit["corrections"],
                )
                for correction in corrections:
                    flatten_edits.append((start_idx, end_idx, correction))
            return flatten_edits

        recall = []
        for true_edits_dl, pred_edits_dl in zip(true_edits_ldl, pred_edits_ldl):
            true_edits_ld = [
                dict(zip(true_edits_dl, t)) for t in zip(*true_edits_dl.values())
            ]
            pred_dicts_ld = [
                dict(zip(pred_edits_dl, t)) for t in zip(*pred_edits_dl.values())
            ]
            gold_flatten_edits = _get_flatten_edits(true_edits_ld)
            pred_flatten_edits = _get_flatten_edits(pred_dicts_ld)
            for gold_flatten_edit in gold_flatten_edits:
                if gold_flatten_edit in pred_flatten_edits:
                    recall.append(1.0)
                else:
                    recall.append(0.0)
        return MetricStats(np.array(recall))


@dataclass
class SeqF1ScoreConfig(F1ScoreConfig):
    tag_schema: str = 'bio'

    def to_metric(self):
        return SeqF1Score(self)


class SeqF1Score(F1Score):
    """
    Calculate F1 score over BIO-tagged spans.
    """

    def calc_stats_from_data(
        self,
        true_data: list[list[str]],
        pred_data: list[list[str]],
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:
        """
        Return sufficient statistics necessary to compute f-score.
        :param true_data: True outputs
        :param pred_data: Predicted outputs
        :param config: Configuration, if over-riding the default
        :return: Returns stats for each class (integer id c) in the following columns of
            MetricStats
            * c*stat_mult + 0: occurrences in the true output
            * c*stat_mult + 1: occurrences in the predicted output
            * c*stat_mult + 2: number of matches with the true output
        """

        # Get span ops
        seq_config = cast(SeqF1ScoreConfig, config or self.config)
        if seq_config.tag_schema == 'bio':
            span_ops: SpanOps = BIOSpanOps()
        elif seq_config.tag_schema == 'bmes':
            span_ops = BMESSpanOps()
        else:
            raise ValueError(f'Illegal tag_schema {seq_config.tag_schema}')

        true_spans_list: list[list[tuple[str, int, int]]] = [
            span_ops.get_spans_simple(true_tags) for true_tags in true_data
        ]
        pred_spans_list: list[list[tuple[str, int, int]]] = [
            span_ops.get_spans_simple(pred_tags) for pred_tags in pred_data
        ]

        # 2. Get tag space
        all_classes = set(
            [
                span[0]
                for span in list(itertools.chain.from_iterable(true_spans_list))
                + list(itertools.chain.from_iterable(pred_spans_list))
            ]
        )
        tag_ids = {k: v for v, k in enumerate([x for x in all_classes])}

        # 3. Create the sufficient statistics
        stat_mult = 3
        n_data, n_classes = len(true_data), len(tag_ids)
        # This is a bit memory inefficient if there's a large number of classes
        stats = np.zeros((n_data, n_classes * stat_mult))

        for i, (true_spans, pred_spans) in enumerate(
            zip(true_spans_list, pred_spans_list)
        ):
            matched_spans = set(true_spans).intersection(pred_spans)
            for offset, spans in enumerate((true_spans, pred_spans, matched_spans)):
                for span in spans:
                    c = tag_ids[span[0]]
                    stats[i, c * stat_mult + offset] += 1
        return MetricStats(stats)


@dataclass
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
        return MetricStats(
            np.array([1.0 / rank for rank in rank_data if rank is not None])
        )


@dataclass
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
            return np.sum(stats.get_data(), axis=0)
        else:
            return np.mean(stats.get_data(), axis=0)

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        raise NotImplementedError


class ExtractiveQAMetric(Metric):
    """
    An abstract class for extractive QA tasks that measures scores after normalization.
    The actual metric must inherit this class and implement the sample_level_metric()
    function.
    """

    def calc_stats_from_data(
        self,
        true_data: list[Union[str, list[str]]],
        pred_data: list[str],
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:
        true_data = [[x] if isinstance(x, str) else x for x in true_data]
        config = self._get_config(config)
        preprocessor = ExtractiveQAPreprocessor(language=config.source_language)
        return MetricStats(
            np.array(
                [
                    max([self.sample_level_metric(t, p, preprocessor) for t in ts])
                    for ts, p in zip(true_data, pred_data)
                ]
            )
        )

    @abc.abstractmethod
    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ) -> float:
        """
        Calculate a score given a ground truth answer string and a prediction.
        """
        ...


@dataclass
class ExactMatchQAConfig(MetricConfig):
    def to_metric(self):
        return ExactMatchQA(self)


class ExactMatchQA(ExtractiveQAMetric):
    """
    Calculate a score for extractive QA based on exact match.
    """

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ) -> float:
        return 1.0 if preprocessor(prediction) == preprocessor(ground_truth) else 0.0


@dataclass
class F1ScoreQAConfig(MetricConfig):
    def to_metric(self):
        return F1ScoreQA(self)


class F1ScoreQA(ExtractiveQAMetric):
    """
    Calculate a score for extractive QA based on F1 score.
    """

    def sample_level_metric(
        self, ground_truth: str, prediction: str, preprocessor: Preprocessor
    ):
        prediction_tokens = preprocessor(prediction).split()
        ground_truth_tokens = preprocessor(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


@dataclass
class LogProbConfig(MetricConfig):
    # If false, return log probability, if true return perplexity
    ppl: bool = False

    def to_metric(self):
        return LogProb(self)


class LogProb(Metric):
    """
    Calculate the log probability
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """
        Take in a list of floats (token-level), or list of lists of floats (sentence
        level) and either one float for each or float+length rows
        """
        if len(pred_data) == 0 or isinstance(pred_data[0], float):
            return MetricStats(np.array(pred_data))
        elif isinstance(pred_data[0], list):
            return MetricStats(np.array([[sum(x), len(x)] for x in pred_data]))
        else:
            t = type(pred_data[0])
            raise ValueError(f'Invalid type of pred_data for calc_stats_from_data {t}')

    def calc_metric_from_aggregate(
        self, agg_stats: np.ndarray, config: Optional[MetricConfig] = None
    ) -> float:
        """From aggregated sufficient statistics, calculate the metric value
        :param agg_stats: aggregated statistics
        :param config: a configuration to over-ride the default for this object
        :return: a single scalar metric value
        """
        config = cast(LogProbConfig, self._get_config(config))
        val = (
            float(agg_stats)
            if (isinstance(agg_stats, float) or agg_stats.size == 1)
            else agg_stats[0] / agg_stats[1]
        )
        if config.ppl:
            val = np.exp(-val)
        return val


def metric_name_to_config(
    name: str, source_language: str, target_language: str
) -> MetricConfig:
    try:
        metric_module = sys.modules[__name__]
        metric_config = getattr(metric_module, f'{name}Config')
        return metric_config(
            name=name, source_language=source_language, target_language=target_language
        )
    except AttributeError:
        if name in EndpointConfig().valid_metrics:
            return EaaSMetricConfig(
                name=name,
                source_language=source_language,
                target_language=target_language,
            )
        else:
            raise ValueError(f'Invalid metric {name}')
