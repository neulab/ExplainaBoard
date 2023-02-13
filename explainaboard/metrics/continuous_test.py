"""Tests for explainaboard.metrics.continuous"""

from __future__ import annotations

import math
import unittest

from explainaboard.metrics.continuous import (
    AbsoluteError,
    AbsoluteErrorConfig,
    RootMeanSquaredError,
    RootMeanSquaredErrorConfig,
)
from explainaboard.metrics.metric import Score
from explainaboard.utils.typing_utils import narrow


class RootMeanSquaredErrorConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            RootMeanSquaredErrorConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
                "negative": False,
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


class RootMeanSquaredErrorTest(unittest.TestCase):
    def test_basic_calculation(self) -> None:
        absolute_error = narrow(
            RootMeanSquaredError, RootMeanSquaredErrorConfig().to_metric()
        )
        expected = [1.0, 3.0]
        actual = [1.4, 2.8]
        metric_result = absolute_error.evaluate(expected, actual)
        value = metric_result.get_value(Score, "score").value
        self.assertAlmostEqual(value, math.sqrt(0.1))

    def test_negative(self) -> None:
        absolute_error = narrow(
            RootMeanSquaredError, RootMeanSquaredErrorConfig(negative=True).to_metric()
        )
        expected = [1.0, 3.0]
        actual = [1.4, 2.8]
        metric_result = absolute_error.evaluate(expected, actual)
        value = metric_result.get_value(Score, "score").value
        self.assertAlmostEqual(value, -math.sqrt(0.1))


class AbsoluteErrorConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            AbsoluteErrorConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
                "negative": False,
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


class AbsoluteErrorTest(unittest.TestCase):
    def test_basic_calculation(self) -> None:
        absolute_error = narrow(AbsoluteError, AbsoluteErrorConfig().to_metric())
        expected = [1.0, 2.0, 3.0]
        actual = [1.5, 2.0, 3.7]
        metric_result = absolute_error.evaluate(expected, actual)
        value = metric_result.get_value(Score, "score").value
        self.assertAlmostEqual(value, 0.4)

    def test_negative(self) -> None:
        absolute_error = narrow(
            AbsoluteError, AbsoluteErrorConfig(negative=True).to_metric()
        )
        expected = [1.0, 2.0, 3.0]
        actual = [1.5, 2.0, 3.7]
        metric_result = absolute_error.evaluate(expected, actual)
        value = metric_result.get_value(Score, "score").value
        self.assertAlmostEqual(value, -0.4)
