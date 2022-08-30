"""Tests for explainaboard.metrics.ranking"""

from __future__ import annotations
import unittest

from explainaboard.metrics.ranking import (
    Hits,
    HitsConfig,
    MeanReciprocalRank,
    MeanReciprocalRankConfig,
    MeanRank,
    MeanRankConfig,
)


class HitsConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            HitsConfig("Hits", hits_k=5).serialize(),
            {
                "name": "Hits",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
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


class MeanReciprocalRankConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            MeanReciprocalRankConfig("MeanReciprocalRank").serialize(),
            {
                "name": "MeanReciprocalRank",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
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


class MeanRankConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            MeanRankConfig("MeanRank").serialize(),
            {
                "name": "MeanRank",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            MeanRankConfig.deserialize({"name": "MeanRank"}),
            MeanRankConfig("MeanRank"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(MeanRankConfig("MeanRank").to_metric(), MeanRank)
