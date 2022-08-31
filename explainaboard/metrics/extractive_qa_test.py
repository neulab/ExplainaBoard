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
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            ExactMatchQAConfig("ExactMatchQA").to_metric(),
            ExactMatchQA,
        )


class F1ScoreQAConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            F1ScoreQAConfig("F1ScoreQA").to_metric(),
            F1ScoreQA,
        )
