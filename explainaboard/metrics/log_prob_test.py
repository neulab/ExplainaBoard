"""Tests for explainaboard.metrics.log_prob"""

from __future__ import annotations

import unittest

from explainaboard.metrics.log_prob import LogProb, LogProbConfig


class LogProbConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            LogProbConfig(ppl=True).serialize(),
            {
                "source_language": None,
                "target_language": None,
                "ppl": True,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            LogProbConfig.deserialize({"ppl": True}),
            LogProbConfig(ppl=True),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(LogProbConfig().to_metric(), LogProb)
