from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from explainaboard.metrics.metric import Metric, MetricConfig, MetricStats
from explainaboard.metrics.registry import register_metric_config


@dataclass
@register_metric_config
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
@register_metric_config
class CorrectCountConfig(MetricConfig):
    def to_metric(self):
        return CorrectCount(self)


class CorrectCount(Accuracy):
    """
    Calculate the absolute value of correct number
    """

    def is_simple_average(self, stats: MetricStats):
        return False

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:

        return MetricStats(
            np.array(
                [
                    (1.0 if y == x or isinstance(x, list) and y in x else 0.0)
                    for x, y in zip(true_data, pred_data)
                ]
            )
        )

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
            return np.sum(data, axis=-2)


@dataclass
@register_metric_config
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
