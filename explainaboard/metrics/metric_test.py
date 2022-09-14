"""Tests for explainaboard.metrics.metric"""

from __future__ import annotations

import unittest

from explainaboard.metrics.metric import (
    ConfidenceInterval,
    get_serializer,
    MetricConfig,
    MetricResult,
    Score,
)


class ScoreTest(unittest.TestCase):
    def test_serialize(self) -> None:
        value = Score(42.0)
        serialized = {"cls_name": "Score", "value": 42.0}
        self.assertEqual(get_serializer().serialize(value), serialized)

    def test_deserialize(self) -> None:
        value = Score(42.0)
        serialized = {"cls_name": "Score", "value": 42.0}
        self.assertEqual(get_serializer().deserialize(serialized), value)


class ConfidenceIntervalTest(unittest.TestCase):
    def test_serialize(self) -> None:
        value = ConfidenceInterval(1.0, 2.0, 0.5)
        serialized = {
            "cls_name": "ConfidenceInterval",
            "low": 1.0,
            "high": 2.0,
            "alpha": 0.5,
        }
        self.assertEqual(get_serializer().serialize(value), serialized)

    def test_deserialize(self) -> None:
        value = ConfidenceInterval(1.0, 2.0, 0.5)
        serialized = {
            "cls_name": "ConfidenceInterval",
            "low": 1.0,
            "high": 2.0,
            "alpha": 0.5,
        }
        self.assertEqual(get_serializer().deserialize(serialized), value)

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
        result = MetricResult(MetricConfig(name="foo"), {"bar": Score(1.25)})
        serialized = {
            "config": {},
            "values": {
                "foo": {"cls_name": "Score", "value": 1.25},
            },
        }
        self.assertEqual(get_serializer().serialize(result), serialized)

    def test_get_value(self) -> None:
        score = Score(1.0)
        ci = ConfidenceInterval(1.0, 2.0, 0.5)

        result = MetricResult(MetricConfig(name="foo"), {"bar": score, "baz": ci})

        # get_value() should return existing objects.
        self.assertIsNone(result.get_value(Score, "foo"))
        self.assertIs(result.get_value(Score, "bar"), score)
        self.assertIsNone(result.get_value(Score, "baz"))
        self.assertIsNone(result.get_value(ConfidenceInterval, "foo"))
        self.assertIsNone(result.get_value(ConfidenceInterval, "bar"))
        self.assertIs(result.get_value(ConfidenceInterval, "baz"), ci)


class MetricConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        with self.assertRaises(NotImplementedError):
            MetricConfig("foo").to_metric()
