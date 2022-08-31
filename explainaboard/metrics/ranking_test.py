"""Tests for explainaboard.metrics.ranking"""

from __future__ import annotations

import unittest

from explainaboard.metrics.ranking import (
    Hits,
    HitsConfig,
    MeanRank,
    MeanRankConfig,
    MeanReciprocalRank,
    MeanReciprocalRankConfig,
)


class HitsConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(HitsConfig("Hits").to_metric(), Hits)


class MeanReciprocalRankConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            MeanReciprocalRankConfig("MeanReciprocalRank").to_metric(),
            MeanReciprocalRank,
        )


class MeanRankConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(MeanRankConfig("MeanRank").to_metric(), MeanRank)
