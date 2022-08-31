"""Tests for explainaboard.metrics.external_eval"""

from __future__ import annotations

import unittest

from explainaboard.metrics.external_eval import ExternalEval, ExternalEvalConfig


class ExternalEvalConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            ExternalEvalConfig("ExternalEval").to_metric(),
            ExternalEval,
        )
