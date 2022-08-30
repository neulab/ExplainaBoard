"""Tests for explainaboard.metrics.extractive_qa"""

from __future__ import annotations

import unittest

from explainaboard.metrics.extractive_qa import (
    ExactMatchQA,
    ExactMatchQAConfig,
    F1ScoreQA,
    F1ScoreQAConfig,
)


class ExactMatchQAConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            ExactMatchQAConfig("ExactMatchQA").serialize(),
            {
                "name": "ExactMatchQA",
                "source_language": None,
                "target_language": None,
                "cls_name": "ExactMatchQAConfig",
                "external_stats": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            ExactMatchQAConfig.deserialize({"name": "ExactMatchQA"}),
            ExactMatchQAConfig("ExactMatchQA"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            ExactMatchQAConfig("ExactMatchQA").to_metric(),
            ExactMatchQA,
        )


class F1ScoreQAConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            F1ScoreQAConfig("F1ScoreQA").serialize(),
            {
                "name": "F1ScoreQA",
                "source_language": None,
                "target_language": None,
                "cls_name": "F1ScoreQAConfig",
                "external_stats": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            F1ScoreQAConfig.deserialize({"name": "F1ScoreQA"}),
            F1ScoreQAConfig("F1ScoreQA"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            F1ScoreQAConfig("F1ScoreQA").to_metric(),
            F1ScoreQA,
        )
