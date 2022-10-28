"""Evaluation metrics to measure F-score."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import cast

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.serialization import common_registry
from explainaboard.utils.span_utils import (
    BIOSpanOps,
    BMESSpanOps,
    gen_argument_pairs,
    SpanOps,
)
from explainaboard.utils.typing_utils import narrow


@dataclass
@common_registry.register("F1ScoreConfig")
class F1ScoreConfig(MetricConfig):
    """Configuration for F1Score metrics.

    Args:
      average: The averaging method, "micro" or "macro".
      separate_match: Whether to use different match counts for precision and recall.
      ignore_classes: Classes for which we should not calculate precision/recall.
    """

    average: str = "micro"
    separate_match: bool = False
    ignore_classes: list[str] = field(default_factory=list)

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return F1Score(self)


class F1Score(Metric):
    """Calculate F1 score, micro- or macro-averaged over classes.

    The numbers calculated should match sklearn's implementation.
    """

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return False

    def calc_stats_from_data(self, true_data: list, pred_data: list) -> MetricStats:
        """Return sufficient statistics necessary to compute f-score.

        Args:
          true_data: True outputs
          pred_data: Predicted outputs

        Returns:
          Returns stats for each class (integer id c) in the following columns of
          MetricStats
          * c*stat_mult + 0: occurrences in the true output
          * c*stat_mult + 1: occurrences in the predicted output
          * c*stat_mult + 2: number of matches with the true output
          * c*stat_mult + 3: number of matches with the predicted output
          (when self.separate_match=True only)
        """
        config = narrow(F1ScoreConfig, self.config)
        stat_mult: int = 4 if config.separate_match else 3

        id_map: dict[str, int] = {}
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
        return SimpleMetricStats(stats)

    def _calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> np.ndarray:
        """See Metric.calc_metric_from_aggregate."""
        is_batched = agg_stats.ndim != 1
        if not is_batched:
            agg_stats = agg_stats.reshape((1, agg_stats.shape[0]))

        config = cast(F1ScoreConfig, self.config)
        supported_averages = {"micro", "macro"}
        stat_mult: int = 4 if config.separate_match else 3
        if config.average not in supported_averages:
            raise ValueError(f"only {supported_averages} supported for now")

        true = agg_stats[:, 0::stat_mult]
        pred = agg_stats[:, 1::stat_mult]
        true_match = agg_stats[:, 2::stat_mult]
        pred_match = agg_stats[:, stat_mult - 1 :: stat_mult]

        if config.average == "micro":
            true, pred, true_match, pred_match = (
                np.sum(x, axis=1) for x in (true, pred, true_match, pred_match)
            )

        np.seterr(invalid="ignore")
        p = np.where(pred != 0.0, pred_match / pred, 0.0)
        r = np.where(true != 0.0, true_match / true, 0.0)
        f1 = np.where(p + r != 0.0, 2 * p * r / (p + r), 0.0)
        np.seterr(invalid="warn")

        if config.average == "macro":
            f1 = np.mean(f1, axis=1)

        if not is_batched:
            f1 = f1[0]

        return f1


@dataclass
@common_registry.register("APEF1ScoreConfig")
class APEF1ScoreConfig(MetricConfig):
    """Configuration for APEF1Score."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return APEF1Score(self)


class APEF1Score(Metric):
    """Calculate F1 score w.r.t the argument pair extraction task.

    Note that this task is different than common sequence labeling tasks
    (such as NER), For example, this is one example's tags:
    'tags': ['Review-B-5',
      'Review-I-5', 'Review-I-5', 'Review-I-5', 'Review-B-7', 'Review-I-7',
      'Review-I-7', 'Review-B-4', 'Review-B-2', 'Review-B-1', 'Review-B-8',
      'Review-I-8', 'Review-I-8', 'Review-I-8', 'Reply-O', 'Reply-B-3', 'Reply-I-3',
      'Reply-I-3', 'Reply-B-5', 'Reply-I-5', 'Reply-I-5', 'Reply-I-5', 'Reply-B-4']
    where
    (Review-B-5, Review-I-5, Review-I-5, Review-I-5, Reply-B-5, Reply-I-5, Reply-I-5,
     Reply-I-5) is one successful identification.
    """

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return False

    def calc_stats_from_data(
        self, true_data: list[list[str]], pred_data: list[list[str]]
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        stats = []

        for tags, pred_tags in zip(true_data, pred_data):
            gold_spans, pred_spans = cast(
                tuple[set, set], gen_argument_pairs(tags, pred_tags)
            )
            stats.append(
                [len(gold_spans), len(pred_spans), len(gold_spans & pred_spans)]
            )
        return SimpleMetricStats(np.array(stats))

    def _calc_metric_from_aggregate(self, agg_stats: np.ndarray) -> np.ndarray:
        """See Metric._calc_metric_from_aggregate."""
        is_batched = agg_stats.ndim == 2
        if not is_batched:
            agg_stats = agg_stats.reshape((1, -1))
        precision = agg_stats[:, 2] * 1.0 / agg_stats[:, 1]
        recall = agg_stats[:, 2] * 1.0 / agg_stats[:, 0]
        fscore = 2.0 * precision * recall / (precision + recall)
        if not is_batched:
            fscore = fscore[0]
        return fscore


@dataclass
@common_registry.register("SeqF1ScoreConfig")
class SeqF1ScoreConfig(F1ScoreConfig):
    """Configuration for SeqF1Score."""

    tag_schema: str = "bio"

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return SeqF1Score(self)


class SeqF1Score(F1Score):
    """Calculate F1 score over BIO-tagged spans."""

    def calc_stats_from_data(
        self,
        true_data: list[list[str]],
        pred_data: list[list[str]],
    ) -> MetricStats:
        """Return sufficient statistics necessary to compute f-score.

        Args:
            true_data: True outputs
            pred_data: Predicted outputs

        Returns:
            Returns stats for each class (integer id c) in the following columns of
            MetricStats
            * c*stat_mult + 0: occurrences in the true output
            * c*stat_mult + 1: occurrences in the predicted output
            * c*stat_mult + 2: number of matches with the true output
        """
        # Get span ops
        seq_config = narrow(SeqF1ScoreConfig, self.config)
        if seq_config.tag_schema == "bio":
            span_ops: SpanOps = BIOSpanOps()
        elif seq_config.tag_schema == "bmes":
            span_ops = BMESSpanOps()
        else:
            raise ValueError(f"Illegal tag_schema {seq_config.tag_schema}")

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
        return SimpleMetricStats(stats)
