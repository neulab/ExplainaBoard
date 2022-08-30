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
    def test_serialize(self) -> None:
        self.assertEqual(
            F1ScoreConfig(
                "F1Score",
                average="macro",
                separate_match=False,
                ignore_classes=None,
            ).serialize(),
            {
                "name": "F1Score",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
                "average": "macro",
                "separate_match": False,
                "ignore_classes": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            F1ScoreConfig.deserialize(
                {
                    "name": "F1Score",
                    "average": "macro",
                    "separate_match": False,
                    "ignore_classes": None,
                }
            ),
            F1ScoreConfig(
                "F1Score",
                average="macro",
                separate_match=False,
                ignore_classes=None,
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            F1ScoreConfig("F1Score").to_metric(),
            F1Score,
        )


class SeqF1ScoreConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            SeqF1ScoreConfig(
                "SeqF1Score",
                average="macro",
                separate_match=False,
                ignore_classes=None,
                tag_schema="bio",
            ).serialize(),
            {
                "name": "SeqF1Score",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
                "average": "macro",
                "separate_match": False,
                "ignore_classes": None,
                "tag_schema": "bio",
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            SeqF1ScoreConfig.deserialize(
                {
                    "name": "SeqF1Score",
                    "average": "macro",
                    "separate_match": False,
                    "ignore_classes": None,
                    "tag_schema": "bio",
                }
            ),
            SeqF1ScoreConfig(
                "SeqF1Score",
                average="macro",
                separate_match=False,
                ignore_classes=None,
                tag_schema="bio",
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            SeqF1ScoreConfig("SeqF1Score").to_metric(),
            SeqF1Score,
        )
