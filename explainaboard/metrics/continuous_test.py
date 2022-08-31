"""Tests for explainaboard.metrics.continuous"""

from __future__ import annotations

import unittest

from explainaboard.metrics.continuous import (
    AbsoluteError,
    AbsoluteErrorConfig,
    RootMeanSquaredError,
    RootMeanSquaredErrorConfig,
)


class RootMeanSquaredErrorConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            RootMeanSquaredErrorConfig("RootMeanSquaredError").to_metric(),
            RootMeanSquaredError,
        )


class AbsoluteErrorConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            AbsoluteErrorConfig("AbsoluteError").to_metric(),
            AbsoluteError,
        )
