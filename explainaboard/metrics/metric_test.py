"""Tests for explainaboard.metrics.metric"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from explainaboard.metrics.metric import (
    Metric,
    MetricConfig,
    MetricStats,
    SimpleMetricStats,
)
from explainaboard.utils.typing_utils import narrow, unwrap


@dataclasses.dataclass
class _DummyMetricConfig(MetricConfig):
    is_simple_average: bool = True

    def to_metric(self) -> Metric:
        return _DummyMetric(self)


class _DummyMetric(Metric):
    def is_simple_average(self, stats: MetricStats):
        return narrow(_DummyMetricConfig, self.config).is_simple_average

    def calc_stats_from_data(
        self,
        true_data: list,
        pred_data: list,
        config: MetricConfig | None = None,
    ) -> MetricStats:
        raise NotImplementedError


class MetricTest(unittest.TestCase):
    def test_aggregate_stats_1dim(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0]))
        aggregate = metric.aggregate_stats(stats)
        self.assertTrue(np.array_equal(aggregate, np.array([2.0])))

    def test_aggregate_stats_2dim(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        aggregate = metric.aggregate_stats(stats)
        self.assertTrue(np.array_equal(aggregate, np.array([3.0, 4.0])))

    def test_aggregate_stats_2dim_empty(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.zeros((0, 3)))
        aggregate = metric.aggregate_stats(stats)
        self.assertTrue(np.array_equal(aggregate, np.zeros((3,))))

    def test_aggregate_stats_3dim(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(
            np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        )
        aggregate = metric.aggregate_stats(stats)
        self.assertTrue(np.array_equal(aggregate, np.array([[2.0, 3.0], [6.0, 7.0]])))

    def test_aggregate_stats_3dim_empty(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.zeros((2, 0, 3)))
        aggregate = metric.aggregate_stats(stats)
        self.assertTrue(np.array_equal(aggregate, np.zeros((2, 3))))

    def test_calc_metric_from_aggregate_0dim(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        aggregate = np.array(3.0)
        with self.assertRaisesRegex(ValueError, r"^Invalid shape size: \(\)$"):
            metric.calc_metric_from_aggregate(aggregate)

    def test_calc_metric_from_aggregate_1dim(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        aggregate = np.array([3.0])
        result = metric.calc_metric_from_aggregate(aggregate)
        self.assertTrue(np.array_equal(result, np.array(3.0)))

    def test_calc_metric_from_aggregate_1dim_multi(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        aggregate = np.array([3.0, 4.0])
        with self.assertRaisesRegex(ValueError, r"^Multiple aggregates"):
            metric.calc_metric_from_aggregate(aggregate)

    def test_calc_metric_from_aggregate_2dim(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        aggregate = np.array([[1.0], [2.0], [3.0]])
        result = metric.calc_metric_from_aggregate(aggregate)
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0, 3.0])))

    def test_calc_metric_from_aggregate_2dim_multi(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        aggregate = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        with self.assertRaisesRegex(ValueError, r"^Multiple aggregates"):
            metric.calc_metric_from_aggregate(aggregate)

    def test_calc_metric_from_aggregate_3dim(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        aggregate = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
        with self.assertRaisesRegex(ValueError, r"Invalid shape size: \(2, 3, 1\)$"):
            metric.calc_metric_from_aggregate(aggregate)

    def test_calc_confidence_interval_tdist(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        ci = unwrap(metric.calc_confidence_interval(stats, 0.05))
        self.assertAlmostEqual(ci[0], 3.387428953673732)
        self.assertAlmostEqual(ci[1], 3.612571046326268)

    def test_calc_confidence_interval_tdist_multi_agg(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with self.assertRaisesRegex(ValueError, r"^t-test can be applied"):
            metric.calc_confidence_interval(stats, 0.05)

    def test_calc_confidence_interval_bootstrap(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test", is_simple_average=False))
        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        ci = unwrap(metric.calc_confidence_interval(stats, 0.05))
        # NOTE(odashi):
        # The sampler takes only 3 samples for each bootstrap iteration, resulting in
        # very wide confidence interval. This is a limitation of bootstrapping.
        self.assertLess(ci[0], ci[1])
        self.assertGreaterEqual(ci[0], 1.0)
        self.assertLessEqual(ci[1], 6.0)

    def test_calc_confidence_interval_bootstrap_multi_agg(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test", is_simple_average=False))
        stats = SimpleMetricStats(np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]]))
        with self.assertRaisesRegex(ValueError, r"^Multiple aggregates"):
            metric.calc_confidence_interval(stats, 0.05)

    def test_calc_confidence_interval_invalid_alpha(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        with self.assertRaisesRegex(ValueError, r"^Invalid confidence_alpha: -0.125$"):
            self.assertIsNone(metric.calc_confidence_interval(stats, -0.125))
        with self.assertRaisesRegex(ValueError, r"^Invalid confidence_alpha: 0.0$"):
            self.assertIsNone(metric.calc_confidence_interval(stats, 0.0))
        with self.assertRaisesRegex(ValueError, r"^Invalid confidence_alpha: 1.0$"):
            self.assertIsNone(metric.calc_confidence_interval(stats, 1.0))
        with self.assertRaisesRegex(ValueError, r"^Invalid confidence_alpha: 1.125$"):
            self.assertIsNone(metric.calc_confidence_interval(stats, 1.125))

    def test_calc_confidence_interval_single_example(self) -> None:
        for is_single_average in (False, True):
            metric = _DummyMetric(
                _DummyMetricConfig("test", is_simple_average=is_single_average)
            )
            stats = SimpleMetricStats(np.array([[1.0]]))
            self.assertIsNone(metric.calc_confidence_interval(stats, 0.05))
