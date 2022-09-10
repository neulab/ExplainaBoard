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
    def test_to_metric(self) -> None:
        self.assertIsInstance(AccuracyConfig("Accuracy").to_metric(), Accuracy)


class CorrectCountConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            CorrectCountConfig("CorrectCount").to_metric(),
            CorrectCount,
        )


class SeqCorrectCountConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            SeqCorrectCountConfig("SeqCorrectCount").to_metric(),
            SeqCorrectCount,
        )
