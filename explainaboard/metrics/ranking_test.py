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
from explainaboard.utils.typing_utils import unwrap


class HitsConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            HitsConfig("Hits", hits_k=5).serialize(),
            {
                "name": "Hits",
                "source_language": None,
                "target_language": None,
                "hits_k": 5,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            HitsConfig.deserialize({"name": "Hits", "hits_k": 5}),
            HitsConfig("Hits", hits_k=5),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(HitsConfig("Hits").to_metric(), Hits)


class HitsTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = HitsConfig(name='Hits').to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = [['a', 'b'], ['c', 'd'], ['c', 'a'], ['a', 'c'], ['b', 'a'], ['a', 'b']]
        result = metric.evaluate(true, pred, confidence_alpha=0.05)
        self.assertAlmostEqual(
            unwrap(result.get_value(Score, "score")).value, 4.0 / 6.0
        )


class MeanReciprocalRankConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            MeanReciprocalRankConfig("MeanReciprocalRank").serialize(),
            {
                "name": "MeanReciprocalRank",
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            MeanReciprocalRankConfig.deserialize({"name": "MeanReciprocalRank"}),
            MeanReciprocalRankConfig("MeanReciprocalRank"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            MeanReciprocalRankConfig("MeanReciprocalRank").to_metric(),
            MeanReciprocalRank,
        )


class MeanReciprocalRankTest(unittest.TestCase):
    def test_evaluate(self) -> None:
        metric = MeanReciprocalRankConfig(name='MRR').to_metric()
        true = ['a', 'b', 'a', 'b', 'a', 'b']
        pred = [['a', 'b'], ['c', 'd'], ['c', 'a'], ['a', 'c'], ['b', 'a'], ['a', 'b']]
        result = metric.evaluate(true, pred, confidence_alpha=0.05)
        self.assertAlmostEqual(
            unwrap(result.get_value(Score, "score")).value, 2.5 / 6.0
        )


class MeanRankConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            MeanRankConfig("MeanRank").serialize(),
            {
                "name": "MeanRank",
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            MeanRankConfig.deserialize({"name": "MeanRank"}),
            MeanRankConfig("MeanRank"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(MeanRankConfig("MeanRank").to_metric(), MeanRank)
