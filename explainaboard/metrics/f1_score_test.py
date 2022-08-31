"""Tests for explainaboard.metrics.f1_score"""

from __future__ import annotations

import unittest

from explainaboard.metrics.f1_score import (
    F1Score,
    F1ScoreConfig,
    SeqF1Score,
    SeqF1ScoreConfig,
)


class F1ScoreConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            F1ScoreConfig("F1Score").to_metric(),
            F1Score,
        )


class SeqF1ScoreConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            SeqF1ScoreConfig("SeqF1Score").to_metric(),
            SeqF1Score,
        )
