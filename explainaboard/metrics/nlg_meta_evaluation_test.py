"""Tests for explainaboard.metrics.nlg_meta_evaluation"""

from __future__ import annotations

import unittest

from explainaboard.metrics.nlg_meta_evaluation import (
    CorrelationConfig,
    KtauCorrelation,
    KtauCorrelationConfig,
    PearsonCorrelation,
    PearsonCorrelationConfig,
)


class CorrelationConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        with self.assertRaises(NotImplementedError):
            CorrelationConfig("Correlation").to_metric()


class KtauCorrelationConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            KtauCorrelationConfig("KtauCorrelation").to_metric(),
            KtauCorrelation,
        )


class PearsonCorrelationConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            PearsonCorrelationConfig("PearsonCorrelation").to_metric(),
            PearsonCorrelation,
        )
