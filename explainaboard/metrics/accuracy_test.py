"""Tests for explainaboard.metrics.accuracy"""

from __future__ import annotations

import unittest

from explainaboard.metrics.accuracy import (
    Accuracy,
    AccuracyConfig,
    CorrectCount,
    CorrectCountConfig,
    SeqCorrectCount,
    SeqCorrectCountConfig,
)
from explainaboard.metrics.metric import Score
from explainaboard.utils.typing_utils import unwrap


class AccuracyConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            AccuracyConfig("Accuracy").serialize(),
            {
                "name": "Accuracy",
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            AccuracyConfig.deserialize({"name": "Accuracy"}),
            AccuracyConfig("Accuracy"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(AccuracyConfig("Accuracy").to_metric(), Accuracy)


class AccuracyTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = AccuracyConfig(name='Accuracy').to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = ['a', 'b', 'a', 'b', 'b', 'a']
        result = metric.evaluate(true, pred, confidence_alpha=0.05)
        self.assertAlmostEqual(
            unwrap(result.get_value(Score, "score")).value, 2.0 / 3.0
        )


class CorrectCountConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            CorrectCountConfig("CorrectCount").serialize(),
            {
                "name": "CorrectCount",
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            CorrectCountConfig.deserialize({"name": "CorrectCount"}),
            CorrectCountConfig("CorrectCount"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            CorrectCountConfig("CorrectCount").to_metric(),
            CorrectCount,
        )


class CorrectCountTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = CorrectCountConfig(name='CorrectCount').to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = ['a', 'b', 'a', 'b', 'b', 'a']
        result = metric.evaluate(true, pred, confidence_alpha=0.05)
        self.assertAlmostEqual(unwrap(result.get_value(Score, "score")).value, 4)


class SeqCorrectCountConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            SeqCorrectCountConfig("SeqCorrectCount").serialize(),
            {
                "name": "SeqCorrectCount",
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            SeqCorrectCountConfig.deserialize({"name": "SeqCorrectCount"}),
            SeqCorrectCountConfig("SeqCorrectCount"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            SeqCorrectCountConfig("SeqCorrectCount").to_metric(),
            SeqCorrectCount,
        )


class SeqCorrectCountTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = SeqCorrectCountConfig(name='SeqCorrectCount').to_metric()
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
        self.assertAlmostEqual(unwrap(result.get_value(Score, "score")).value, 5)
