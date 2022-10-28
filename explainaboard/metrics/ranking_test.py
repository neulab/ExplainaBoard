"""Tests for explainaboard.metrics.ranking"""

from __future__ import annotations

import unittest

from explainaboard.metrics.metric import Score
from explainaboard.metrics.ranking import (
    Hits,
    HitsConfig,
    MeanRank,
    MeanRankConfig,
    MeanReciprocalRank,
    MeanReciprocalRankConfig,
)


class HitsConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            HitsConfig(hits_k=5).serialize(),
            {
                "source_language": None,
                "target_language": None,
                "hits_k": 5,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            HitsConfig.deserialize({"hits_k": 5}),
            HitsConfig(hits_k=5),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(HitsConfig().to_metric(), Hits)


class HitsTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = HitsConfig().to_metric()
        true = ["a", "b", "a", "b", "a", "b"]
        pred = [["a", "b"], ["c", "d"], ["c", "a"], ["a", "c"], ["b", "a"], ["a", "b"]]
        result = metric.evaluate(true, pred, confidence_alpha=0.05)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 4.0 / 6.0)


class MeanReciprocalRankConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            MeanReciprocalRankConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            MeanReciprocalRankConfig.deserialize({}),
            MeanReciprocalRankConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            MeanReciprocalRankConfig().to_metric(),
            MeanReciprocalRank,
        )


class MeanReciprocalRankTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = MeanReciprocalRankConfig().to_metric()
        true = ["a", "b", "a", "b", "a", "b"]
        pred = [["a", "b"], ["c", "d"], ["c", "a"], ["a", "c"], ["b", "a"], ["a", "b"]]
        result = metric.evaluate(true, pred, confidence_alpha=0.05)
        self.assertAlmostEqual(result.get_value(Score, "score").value, 2.5 / 6.0)


class MeanRankConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            MeanRankConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            MeanRankConfig.deserialize({}),
            MeanRankConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(MeanRankConfig().to_metric(), MeanRank)
