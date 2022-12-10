"""Tests for explainaboard.metrics.metric"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any
import unittest

import numpy as np

from explainaboard.metrics.metric import (
    ConfidenceInterval,
    Metric,
    MetricConfig,
    MetricResult,
    MetricStats,
    Score,
    SimpleMetricStats,
)
from explainaboard.serialization.types import SerializableData
from explainaboard.utils.typing_utils import narrow, unwrap


@dataclasses.dataclass
class _DummyMetricConfig(MetricConfig):
    is_simple_average: bool = True
    uses_customized_aggregate: bool = False
    aggregate_stats_fn: Callable[[MetricStats], np.ndarray[Any, Any]] | None = None

    def to_metric(self) -> Metric:
        return _DummyMetric(self)


class _DummyMetric(Metric):
    def is_simple_average(self, stats: MetricStats) -> bool:
        return narrow(_DummyMetricConfig, self.config).is_simple_average

    def uses_customized_aggregate(self) -> bool:
        return narrow(_DummyMetricConfig, self.config).uses_customized_aggregate

    def _aggregate_stats(
        self, stats: MetricStats
    ) -> np.ndarray[tuple[int], Any] | np.ndarray[tuple[int, int], Any]:
        user_agg_fn = narrow(_DummyMetricConfig, self.config).aggregate_stats_fn
        agg_fn = user_agg_fn if user_agg_fn is not None else super()._aggregate_stats
        return agg_fn(stats)

    def calc_stats_from_data(
        self,
        true_data: list,
        pred_data: list,
        config: MetricConfig | None = None,
    ) -> MetricStats:
        raise NotImplementedError


class ScoreTest(unittest.TestCase):
    def test_serialize(self) -> None:
        value = Score(42.0)
        serialized: dict[str, SerializableData] = {"value": 42.0}
        self.assertEqual(value.serialize(), serialized)

    def test_deserialize(self) -> None:
        value = Score(42.0)
        serialized: dict[str, SerializableData] = {"value": 42.0}
        self.assertEqual(narrow(Score, Score.deserialize(serialized)), value)


class ConfidenceIntervalTest(unittest.TestCase):
    def test_serialize(self) -> None:
        value = ConfidenceInterval(1.0, 2.0, 0.5)
        serialized: dict[str, SerializableData] = {
            "low": 1.0,
            "high": 2.0,
            "alpha": 0.5,
        }
        self.assertEqual(value.serialize(), serialized)

    def test_deserialize(self) -> None:
        value = ConfidenceInterval(1.0, 2.0, 0.5)
        serialized: dict[str, SerializableData] = {
            "low": 1.0,
            "high": 2.0,
            "alpha": 0.5,
        }
        self.assertEqual(
            narrow(ConfidenceInterval, ConfidenceInterval.deserialize(serialized)),
            value,
        )

    def test_invalid_values(self) -> None:
        # low == high is valid.
        ConfidenceInterval(1.0, 1.0, 0.5)

        with self.assertRaisesRegex(ValueError, r"^`high` must be"):
            ConfidenceInterval(1.0, 0.999, 0.5)
        with self.assertRaisesRegex(ValueError, r"^`alpha` must be"):
            ConfidenceInterval(1.0, 2.0, -0.1)
        with self.assertRaisesRegex(ValueError, r"^`alpha` must be"):
            ConfidenceInterval(1.0, 2.0, 0.0)
        with self.assertRaisesRegex(ValueError, r"^`alpha` must be"):
            ConfidenceInterval(1.0, 2.0, 1.0)
        with self.assertRaisesRegex(ValueError, r"^`alpha` must be"):
            ConfidenceInterval(1.0, 2.0, 1.1)


class MetricResultTest(unittest.TestCase):
    def test_serialize(self) -> None:
        score = Score(1.0)
        ci = ConfidenceInterval(1.0, 2.0, 0.5)
        result = MetricResult({"bar": score, "baz": ci})
        serialized: dict[str, SerializableData] = {
            "values": {"bar": score, "baz": ci},
        }
        self.assertEqual(result.serialize(), serialized)

    def test_deserialize(self) -> None:
        score = Score(1.0)
        ci = ConfidenceInterval(1.0, 2.0, 0.5)
        config = _DummyMetricConfig()
        serialized: dict[str, SerializableData] = {
            "config": config,
            "values": {"bar": score, "baz": ci},
        }
        restored = narrow(MetricResult, MetricResult.deserialize(serialized))
        self.assertEqual(restored._values, {"bar": score, "baz": ci})

    def test_eq(self) -> None:
        s1 = Score(1.0)
        s2 = Score(1.0)
        s3 = Score(2.0)
        c1 = ConfidenceInterval(1.0, 2.0, 0.5)
        c2 = ConfidenceInterval(1.0, 2.0, 0.5)
        c3 = ConfidenceInterval(1.0, 2.0, 0.6)

        r = MetricResult

        self.assertEqual(r({"a": s1}), r({"a": s2}))
        self.assertEqual(r({"a": c1}), r({"a": c2}))
        self.assertEqual(r({"a": s1, "b": c1}), r({"a": s2, "b": c2}))

        # Different keys
        self.assertNotEqual(r({"a": s1}), r({"b": s1}))
        self.assertNotEqual(r({"a": s1, "b": c1}), r({"a": s1}))
        self.assertNotEqual(r({"a": s1, "b": c1}), r({"a": s1, "c": c1}))

        # Different types
        self.assertNotEqual(r({"a": s1}), r({"a": c1}))
        self.assertNotEqual(r({"a": s1, "b": c1}), r({"a": s1, "b": s2}))

        # Different values
        self.assertNotEqual(r({"a": s1}), r({"a": s3}))
        self.assertNotEqual(r({"a": s1, "b": c1}), r({"a": s1, "b": c3}))

    def test_get_value(self) -> None:
        score = Score(1.0)
        ci = ConfidenceInterval(1.0, 2.0, 0.5)

        result = MetricResult({"bar": score, "baz": ci})

        # get_value() should return existing objects.
        with self.assertRaisesRegex(ValueError, r"^MetricValue \"foo\" not found.$"):
            result.get_value(Score, "foo")
        self.assertIs(result.get_value(Score, "bar"), score)
        with self.assertRaisesRegex(
            ValueError, r"^MetricValue \"baz\" is not a subclass of Score.$"
        ):
            result.get_value(Score, "baz")
        with self.assertRaisesRegex(ValueError, r"^MetricValue \"foo\" not found.$"):
            result.get_value(ConfidenceInterval, "foo")
        with self.assertRaisesRegex(
            ValueError,
            r"^MetricValue \"bar\" is not a subclass of ConfidenceInterval.$",
        ):
            result.get_value(ConfidenceInterval, "bar")
        self.assertIs(result.get_value(ConfidenceInterval, "baz"), ci)

    def test_get_value_or_none(self) -> None:
        score = Score(1.0)
        ci = ConfidenceInterval(1.0, 2.0, 0.5)

        result = MetricResult({"bar": score, "baz": ci})

        # get_value_or_none() should return existing objects.
        self.assertIsNone(result.get_value_or_none(Score, "foo"))
        self.assertIs(result.get_value_or_none(Score, "bar"), score)
        self.assertIsNone(result.get_value_or_none(Score, "baz"))
        self.assertIsNone(result.get_value_or_none(ConfidenceInterval, "foo"))
        self.assertIsNone(result.get_value_or_none(ConfidenceInterval, "bar"))
        self.assertIs(result.get_value_or_none(ConfidenceInterval, "baz"), ci)


class MetricConfigTest(unittest.TestCase):
    def test_replace_languages(self) -> None:
        config = _DummyMetricConfig(source_language="xx", target_language="yy")
        new_config = config.replace_languages(
            source_language="aa", target_language="bb"
        )
        self.assertIsNot(new_config, config)
        self.assertEqual(config.source_language, "xx")
        self.assertEqual(config.target_language, "yy")
        self.assertEqual(new_config.source_language, "aa")
        self.assertEqual(new_config.target_language, "bb")


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

    def test_aggregate_stats_customized_nonbatch(self) -> None:
        def agg_fn(stats: MetricStats) -> np.ndarray[Any, Any]:
            return stats.get_data().max(axis=-2).max(axis=-1, keepdims=True)

        metric = _DummyMetric(
            _DummyMetricConfig(
                "test", uses_customized_aggregate=True, aggregate_stats_fn=agg_fn
            )
        )
        stats = SimpleMetricStats(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        aggregate = metric.aggregate_stats(stats)
        self.assertTrue(np.array_equal(aggregate, np.array([6.0])))

    def test_aggregate_stats_customized_nonbatch_invalid(self) -> None:
        def agg_fn(stats: MetricStats) -> np.ndarray[Any, Any]:
            return stats.get_data().max(axis=-2).max(axis=-1, keepdims=True)

        metric = _DummyMetric(_DummyMetricConfig("test", aggregate_stats_fn=agg_fn))
        stats = SimpleMetricStats(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        with self.assertRaisesRegex(
            AssertionError, r"Expected shape \(2,\), but got \(1,\)\.$"
        ):
            metric.aggregate_stats(stats)

    def test_aggregate_stats_customized_batch(self) -> None:
        def agg_fn(stats: MetricStats) -> np.ndarray[Any, Any]:
            return stats.get_batch_data().max(axis=-2).max(axis=-1, keepdims=True)

        metric = _DummyMetric(
            _DummyMetricConfig(
                "test", uses_customized_aggregate=True, aggregate_stats_fn=agg_fn
            )
        )
        stats = SimpleMetricStats(
            np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        )
        aggregate = metric.aggregate_stats(stats)
        self.assertTrue(np.array_equal(aggregate, np.array([[4.0], [8.0]])))

    def test_aggregate_stats_customized_batch_invalid(self) -> None:
        def agg_fn(stats: MetricStats) -> np.ndarray[Any, Any]:
            return stats.get_batch_data().max(axis=-2).max(axis=-1, keepdims=True)

        metric = _DummyMetric(_DummyMetricConfig("test", aggregate_stats_fn=agg_fn))
        stats = SimpleMetricStats(
            np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        )
        with self.assertRaisesRegex(
            AssertionError, r"Expected shape \(2, 2\), but got \(2, 1\)\.$"
        ):
            metric.aggregate_stats(stats)

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
        stats = SimpleMetricStats(np.arange(1.0, 31.0))
        ci = unwrap(metric.calc_confidence_interval(stats, 0.05))
        self.assertAlmostEqual(ci[0], 12.268005046817326)
        self.assertAlmostEqual(ci[1], 18.731994953182674)

    def test_calc_confidence_interval_tdist_multi_agg(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.arange(1, 61).reshape(30, 2))
        with self.assertRaisesRegex(ValueError, r"^t-test can be applied"):
            metric.calc_confidence_interval(stats, 0.05)

    def test_calc_confidence_interval_bootstrap(self) -> None:
        metric = _DummyMetric(
            _DummyMetricConfig("test", is_simple_average=False),
            seed=np.random.SeedSequence(12345),
        )
        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        ci = unwrap(metric.calc_confidence_interval(stats, 0.05))
        # NOTE(odashi):
        # The sampler takes only 3 samples for each bootstrap iteration, resulting in
        # very wide confidence interval. This is a limitation of bootstrapping.
        self.assertAlmostEqual(ci[0], 2.166666666666666)
        self.assertAlmostEqual(ci[1], 4.833333333333333)

    def test_calc_confidence_interval_bootstrap_multi_agg(self) -> None:
        metric = _DummyMetric(
            _DummyMetricConfig("test", is_simple_average=False),
            seed=np.random.SeedSequence(12345),
        )
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

    def test_evaluate_from_stats_tdist_without_ci(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = metric.evaluate_from_stats(stats, confidence_alpha=None)
        self.assertEqual(result.get_value(Score, "score").value, 3.0)
        self.assertIsNone(result.get_value_or_none(ConfidenceInterval, "score_ci"))

    def test_evaluate_from_stats_tdist_with_ci(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.arange(1.0, 31.0))
        result = metric.evaluate_from_stats(stats, confidence_alpha=0.05)
        self.assertEqual(result.get_value(Score, "score").value, 15.5)
        ci = result.get_value(ConfidenceInterval, "score_ci")
        self.assertAlmostEqual(ci.low, 12.268005046817326)
        self.assertAlmostEqual(ci.high, 18.731994953182674)

    def test_evaluate_from_stats_tdist_single_data(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test"))
        stats = SimpleMetricStats(np.array([3.0]))
        result = metric.evaluate_from_stats(stats, confidence_alpha=0.05)
        self.assertEqual(result.get_value(Score, "score").value, 3.0)
        self.assertIsNone(result.get_value_or_none(ConfidenceInterval, "score_ci"))

    def test_evaluate_from_stats_bootstrap_without_ci(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test", is_simple_average=False))
        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = metric.evaluate_from_stats(stats, confidence_alpha=None)
        self.assertEqual(result.get_value(Score, "score").value, 3.0)
        self.assertIsNone(result.get_value_or_none(ConfidenceInterval, "score_ci"))

    def test_evaluate_from_stats_bootstrap_with_ci(self) -> None:
        metric = _DummyMetric(
            _DummyMetricConfig("test", is_simple_average=False),
            seed=np.random.SeedSequence(12345),
        )
        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = metric.evaluate_from_stats(stats, confidence_alpha=0.05)
        self.assertEqual(result.get_value(Score, "score").value, 3.0)
        ci = result.get_value(ConfidenceInterval, "score_ci")
        # TODO(odahsi): According to the current default settings of bootstrapping,
        # estimated confidence intervals tends to become very wide for small data
        self.assertAlmostEqual(ci.low, 1.8)
        self.assertAlmostEqual(ci.high, 4.2)

    def test_evaluate_from_stats_bootstrap_single_data(self) -> None:
        metric = _DummyMetric(_DummyMetricConfig("test", is_simple_average=False))
        stats = SimpleMetricStats(np.array([3.0]))
        result = metric.evaluate_from_stats(stats, confidence_alpha=0.05)
        self.assertEqual(result.get_value(Score, "score").value, 3.0)
        self.assertIsNone(result.get_value_or_none(ConfidenceInterval, "score_ci"))
