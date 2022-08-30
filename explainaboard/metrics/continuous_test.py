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
            RootMeanSquaredErrorConfig("RootMeanSquaredError").serialize(),
            {
                "name": "RootMeanSquaredError",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            RootMeanSquaredErrorConfig.deserialize(
                {"name": "RootMeanSquaredError"},
            ),
            RootMeanSquaredErrorConfig("RootMeanSquaredError"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            RootMeanSquaredErrorConfig("RootMeanSquaredError").to_metric(),
            RootMeanSquaredError,
        )


class AbsoluteErrorConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            AbsoluteErrorConfig("AbsoluteError").serialize(),
            {
                "name": "AbsoluteError",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            AbsoluteErrorConfig.deserialize({"name": "AbsoluteError"}),
            AbsoluteErrorConfig("AbsoluteError"),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            AbsoluteErrorConfig("AbsoluteError").to_metric(),
            AbsoluteError,
        )
