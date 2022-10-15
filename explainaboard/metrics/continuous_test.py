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
    def test_serialize(self) -> None:
        self.assertEqual(
            RootMeanSquaredErrorConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            RootMeanSquaredErrorConfig.deserialize({}),
            RootMeanSquaredErrorConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            RootMeanSquaredErrorConfig().to_metric(),
            RootMeanSquaredError,
        )


class AbsoluteErrorConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            AbsoluteErrorConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            AbsoluteErrorConfig.deserialize({}),
            AbsoluteErrorConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            AbsoluteErrorConfig().to_metric(),
            AbsoluteError,
        )
