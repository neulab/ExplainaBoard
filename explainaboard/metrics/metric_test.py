"""Tests for explainaboard.metrics.metric"""

from __future__ import annotations

import dataclasses
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


class DummyMetricConfig(MetricConfig):
    def to_metric(self) -> Metric:
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
        config = DummyMetricConfig(name="foo")
        result = MetricResult(config, {"bar": score, "baz": ci})
        serialized: dict[str, SerializableData] = {
            "config": config,
            "values": {"bar": score, "baz": ci},
        }
        self.assertEqual(result.serialize(), serialized)

    def test_deserialize(self) -> None:
        score = Score(1.0)
        ci = ConfidenceInterval(1.0, 2.0, 0.5)
        config = DummyMetricConfig(name="foo")
        serialized: dict[str, SerializableData] = {
            "config": config,
            "values": {"bar": score, "baz": ci},
        }
        restored = narrow(MetricResult, MetricResult.deserialize(serialized))
        self.assertIs(restored.config, config)
        self.assertEqual(restored._values, {"bar": score, "baz": ci})

    def test_get_value(self) -> None:
        score = Score(1.0)
        ci = ConfidenceInterval(1.0, 2.0, 0.5)

        result = MetricResult(DummyMetricConfig(name="foo"), {"bar": score, "baz": ci})

        # get_value() should return existing objects.
        self.assertIsNone(result.get_value(Score, "foo"))
        self.assertIs(result.get_value(Score, "bar"), score)
        self.assertIsNone(result.get_value(Score, "baz"))
        self.assertIsNone(result.get_value(ConfidenceInterval, "foo"))
        self.assertIsNone(result.get_value(ConfidenceInterval, "bar"))
        self.assertIs(result.get_value(ConfidenceInterval, "baz"), ci)


@dataclasses.dataclass
class _DummyMetricConfig(MetricConfig):
    is_simple_average: bool = False

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
    def test_evaluate_from_stats(self) -> None:
        config = _DummyMetricConfig("test", is_simple_average=True)
        metric = _DummyMetric(config)

        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        result = metric.evaluate_from_stats(stats, confidence_alpha=None)
        self.assertEqual(unwrap(result.get_value(Score, "score")).value, 3.0)
        self.assertIsNone(result.get_value(ConfidenceInterval, "score_ci"))

        result = metric.evaluate_from_stats(stats, confidence_alpha=0.05)
        self.assertEqual(unwrap(result.get_value(Score, "score")).value, 3.0)
        ci = unwrap(result.get_value(ConfidenceInterval, "score_ci"))
        self.assertGreater(ci.low, 2.8)
        self.assertLess(ci.high, 3.2)

        stats = SimpleMetricStats(np.array([3.0]))

        result = metric.evaluate_from_stats(stats, confidence_alpha=0.05)
        self.assertEqual(unwrap(result.get_value(Score, "score")).value, 3.0)
        self.assertIsNone(result.get_value(ConfidenceInterval, "score_ci"))

    def test_evaluate_from_stats_bootstrap(self) -> None:
        config = _DummyMetricConfig("test", is_simple_average=False)
        metric = _DummyMetric(config)

        stats = SimpleMetricStats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        result = metric.evaluate_from_stats(stats, confidence_alpha=None)
        self.assertEqual(unwrap(result.get_value(Score, "score")).value, 3.0)
        self.assertIsNone(result.get_value(ConfidenceInterval, "score_ci"))

        result = metric.evaluate_from_stats(stats, confidence_alpha=0.05)
        self.assertEqual(unwrap(result.get_value(Score, "score")).value, 3.0)
        ci = unwrap(result.get_value(ConfidenceInterval, "score_ci"))
        print(dataclasses.asdict(ci))
        # TODO(odahsi): According to the current default settings of bootstrapping,
        # estimated confidence intervals tends to become very wide for small data
        self.assertGreaterEqual(ci.low, 1.0)
        self.assertLessEqual(ci.high, 5.0)

        stats = SimpleMetricStats(np.array([3.0]))

        result = metric.evaluate_from_stats(stats, confidence_alpha=0.05)
        self.assertEqual(unwrap(result.get_value(Score, "score")).value, 3.0)
        self.assertIsNone(result.get_value(ConfidenceInterval, "score_ci"))
