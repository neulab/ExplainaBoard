"""Tests for explainaboard.metrics.metric"""

from __future__ import annotations

import unittest

from explainaboard.metrics.metric import (
    ConfidenceInterval,
    Metric,
    MetricConfig,
    MetricResult,
    Score,
)
from explainaboard.serialization.types import SerializableData
from explainaboard.utils.typing_utils import narrow


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
        with self.assertRaisesRegex(ValueError, r"^`high` must be"):
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
