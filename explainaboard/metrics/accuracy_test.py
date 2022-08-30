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


class AccuracyConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            AccuracyConfig("Accuracy").serialize(),
            {
                "name": "Accuracy",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            AccuracyConfig.deserialize({"name": "Accuracy"}),
            AccuracyConfig("Accuracy"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(AccuracyConfig("Accuracy").to_metric(), Accuracy)


class CorrectCountConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            CorrectCountConfig("CorrectCount").serialize(),
            {
                "name": "CorrectCount",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
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


class SeqCorrectCountConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            SeqCorrectCountConfig("SeqCorrectCount").serialize(),
            {
                "name": "SeqCorrectCount",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
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
