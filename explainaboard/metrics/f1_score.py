from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import cast, Optional

import numpy as np

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config
from explainaboard.utils.span_utils import BIOSpanOps, BMESSpanOps, SpanOps
from explainaboard.utils.typing_utils import unwrap_or


@dataclass
@register_metric_config
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

    def is_simple_average(self, stats: MetricStats):
        return False

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
        config = cast(F1ScoreConfig, unwrap_or(config, self.config))
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
    ) -> np.ndarray:

        if agg_stats.size == 1:
            return agg_stats

        if agg_stats.ndim == 1:
            agg_stats = agg_stats.reshape((1, agg_stats.shape[0]))

        config = cast(F1ScoreConfig, unwrap_or(config, self.config))
        supported_averages = {'micro', 'macro'}
        stat_mult: int = 4 if config.separate_match else 3
        if config.average not in supported_averages:
            raise ValueError(f'only {supported_averages} supported for now')

        true = agg_stats[:, 0::stat_mult]
        pred = agg_stats[:, 1::stat_mult]
        true_match = agg_stats[:, 2::stat_mult]
        pred_match = agg_stats[:, stat_mult - 1 :: stat_mult]

        if config.average == 'micro':
            true, pred, true_match, pred_match = (
                np.sum(x, axis=1) for x in (true, pred, true_match, pred_match)
            )

        p = np.where(pred != 0.0, pred_match / pred, 0.0)
        r = np.where(true != 0.0, true_match / true, 0.0)
        f1 = np.where(p + r != 0.0, 2 * p * r / (p + r), 0.0)

        if config.average == 'macro':
            f1 = np.mean(f1, axis=1)

        return f1


@dataclass
@register_metric_config
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
