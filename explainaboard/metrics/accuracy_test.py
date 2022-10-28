"""Tests for explainaboard.metrics.accuracy"""

from __future__ import annotations

import unittest

import numpy as np

from explainaboard.metrics.accuracy import (
    Accuracy,
    AccuracyConfig,
    CorrectCount,
    CorrectCountConfig,
    SeqCorrectCount,
    SeqCorrectCountConfig,
)
from explainaboard.metrics.metric import Score, SimpleMetricStats


class AccuracyConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            AccuracyConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            AccuracyConfig.deserialize({}),
            AccuracyConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(AccuracyConfig().to_metric(), Accuracy)


class AccuracyTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = AccuracyConfig().to_metric()
        true = ["a", "b", "a", "b", "a", "b"]
        pred = ["a", "b", "a", "b", "b", "a"]
        result = metric.evaluate(true, pred, confidence_alpha=0.05)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 2.0 / 3.0)

    def test_evaluate_from_stats(self) -> None:
        metric = AccuracyConfig().to_metric()
        stats = SimpleMetricStats(np.array([1.0, 0.0, 1.0, 1.0, 0.0]))
        result = metric.evaluate_from_stats(stats=stats)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 0.6)
        self.assertIsNone(result.get_value_or_none(Score, "confidence"))

    def test_evaluate_from_stats_with_confidence(self) -> None:
        metric = AccuracyConfig().to_metric()
        stats = SimpleMetricStats(np.array([1.0, 0.0, 1.0, 1.0, 0.0]))
        conf_stats = SimpleMetricStats(np.array([0.5, 0.4, 0.6, 0.5, 0.5]))
        result = metric.evaluate_from_stats(stats=stats, auxiliary_stats=conf_stats)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 0.6)
        self.assertAlmostEqual(result.get_value(Score, "confidence").value, 0.5)


class CorrectCountConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            CorrectCountConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            CorrectCountConfig.deserialize({}),
            CorrectCountConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            CorrectCountConfig().to_metric(),
            CorrectCount,
        )


class CorrectCountTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = CorrectCountConfig().to_metric()
        true = ["a", "b", "a", "b", "a", "b"]
        pred = ["a", "b", "a", "b", "b", "a"]
        result = metric.evaluate(true, pred, confidence_alpha=0.05)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 4)


class SeqCorrectCountConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            SeqCorrectCountConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            SeqCorrectCountConfig.deserialize({}),
            SeqCorrectCountConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            SeqCorrectCountConfig().to_metric(),
            SeqCorrectCount,
        )


class SeqCorrectCountTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = SeqCorrectCountConfig().to_metric()
        true = [
            {
                "start_idx": [8, 17, 39, 46, 58, 65, 65, 80],
                "end_idx": [8, 18, 40, 47, 59, 65, 66, 81],
                "corrections": [
                    ["the"],
                    ["found"],
                    ["other"],
                    ["there"],
                    ["chickens."],
                    ["in"],
                    ["which"],
                    ["selling"],
                ],
            }
        ]
        pred = [
            {
                "start_idx": [8, 17, 39, 46, 58],
                "end_idx": [8, 18, 40, 47, 59],
                "corrections": [
                    ["the"],
                    ["found"],
                    ["other"],
                    ["there"],
                    ["chickens."],
                ],
            }
        ]
        result = metric.evaluate(true, pred)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 5)
