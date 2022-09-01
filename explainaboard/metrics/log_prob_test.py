"""Tests for explainaboard.metrics.log_prob"""

from __future__ import annotations

import unittest

from explainaboard.metrics.log_prob import LogProb, LogProbConfig


class LogProbConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(LogProbConfig("LogProb").to_metric(), LogProb)
