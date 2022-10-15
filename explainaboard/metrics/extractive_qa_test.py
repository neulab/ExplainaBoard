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
            ExactMatchQAConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            ExactMatchQAConfig.deserialize({}),
            ExactMatchQAConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            ExactMatchQAConfig().to_metric(),
            ExactMatchQA,
        )


class F1ScoreQAConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            F1ScoreQAConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            F1ScoreQAConfig.deserialize({}),
            F1ScoreQAConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            F1ScoreQAConfig().to_metric(),
            F1ScoreQA,
        )
