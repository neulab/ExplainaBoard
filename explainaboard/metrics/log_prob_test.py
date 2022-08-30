"""Tests for explainaboard.metrics.log_prob"""

from __future__ import annotations
import unittest

from explainaboard.metrics.log_prob import LogProb, LogProbConfig


class LogProbConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            LogProbConfig("LogProb", ppl=True).serialize(),
            {
                "name": "LogProb",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
                "ppl": True,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            LogProbConfig.deserialize({"name": "LogProb", "ppl": True}),
            LogProbConfig("LogProb", ppl=True),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(LogProbConfig("LogProb").to_metric(), LogProb)
