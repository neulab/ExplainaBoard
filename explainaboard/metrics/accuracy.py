"""Evaluation metrics measuring accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.metrics.registry import metric_config_registry


@dataclass
@metric_config_registry.register("AccuracyConfig")
class AccuracyConfig(MetricConfig):
    """Configuration for the Accuracy metric."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return Accuracy(self)


class Accuracy(Metric):
    """Calculate zero-one accuracy.

    The score is 1 iff the prediction equals the ground truth.
    """

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        return SimpleMetricStats(
            np.array(
                [
                    (1.0 if y == x or isinstance(x, list) and y in x else 0.0)
                    for x, y in zip(true_data, pred_data)
                ]
            )
        )


@dataclass
@metric_config_registry.register("CorrectCountConfig")
class CorrectCountConfig(MetricConfig):
    """Configuration for CorrectCount."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return CorrectCount(self)


class CorrectCount(Accuracy):
    """Calculate the absolute value of correct answers."""

    def is_simple_average(self, stats: MetricStats):
        """See Metric.is_simple_average."""
        return False

    def calc_stats_from_data(
        self, true_data: list, pred_data: list, config: Optional[MetricConfig] = None
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        return SimpleMetricStats(
            np.array(
                [
                    (1.0 if y == x or isinstance(x, list) and y in x else 0.0)
                    for x, y in zip(true_data, pred_data)
                ]
            )
        )

    def aggregate_stats(self, stats: MetricStats) -> np.ndarray:
        """See Metric.aggregate_stats."""
        data = stats.get_batch_data() if stats.is_batched() else stats.get_data()
        if data.size == 0:
            return np.array(0.0)
        else:
            return np.sum(data, axis=-2)


@dataclass
@metric_config_registry.register("SeqCorrectCountConfig")
class SeqCorrectCountConfig(MetricConfig):
    """Configuration for SeqCorrectCount."""

    def to_metric(self) -> Metric:
        """See MetricConfig.to_metric."""
        return SeqCorrectCount(self)


class SeqCorrectCount(CorrectCount):
    """A count of the total number of times a sequence is correct."""

    @staticmethod
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

    def calc_stats_from_data(
        self,
        true_edits_ldl: list[dict[str, list]],
        pred_edits_ldl: list[dict[str, list]],
        config: Optional[MetricConfig] = None,
    ) -> MetricStats:
        """See Metric.calc_stats_from_data."""
        recall = []
        for true_edits_dl, pred_edits_dl in zip(true_edits_ldl, pred_edits_ldl):
            true_edits_ld = [
                dict(zip(true_edits_dl, t)) for t in zip(*true_edits_dl.values())
            ]
            pred_dicts_ld = [
                dict(zip(pred_edits_dl, t)) for t in zip(*pred_edits_dl.values())
            ]
            gold_flatten_edits = self._get_flatten_edits(true_edits_ld)
            pred_flatten_edits = self._get_flatten_edits(pred_dicts_ld)
            for gold_flatten_edit in gold_flatten_edits:
                if gold_flatten_edit in pred_flatten_edits:
                    recall.append(1.0)
                else:
                    recall.append(0.0)
        return SimpleMetricStats(np.array(recall))
