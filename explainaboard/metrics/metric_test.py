"""Tests for explainaboard.metrics.metric"""

from __future__ import annotations

import unittest

from explainaboard.metrics.metric import MetricConfig


class MetricConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        with self.assertRaises(NotImplementedError):
            MetricConfig("foo").to_metric()
