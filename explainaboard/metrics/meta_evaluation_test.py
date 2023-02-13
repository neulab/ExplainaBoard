"""Tests for explainaboard.metrics.meta_evaluation"""

from __future__ import annotations

import math
import unittest

from explainaboard.metrics.meta_evaluation import (
    AbsoluteErrorMetaEval,
    AbsoluteErrorMetaEvalConfig,
    RootMeanSquaredErrorMetaEval,
    RootMeanSquaredErrorMetaEvalConfig,
)
from explainaboard.metrics.metric import Score
from explainaboard.utils.typing_utils import narrow


class RootMeanSquaredErrorMetaEvalConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            RootMeanSquaredErrorMetaEvalConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
                "negative": False,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            RootMeanSquaredErrorMetaEvalConfig.deserialize({}),
            RootMeanSquaredErrorMetaEvalConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            RootMeanSquaredErrorMetaEvalConfig().to_metric(),
            RootMeanSquaredErrorMetaEval,
        )


class RootMeanSquaredErrorMetaEvalTest(unittest.TestCase):
    def test_basic_calculation(self) -> None:
        absolute_error = narrow(
            RootMeanSquaredErrorMetaEval,
            RootMeanSquaredErrorMetaEvalConfig().to_metric(),
        )
        expected = [[1.0, 3.0, 1.0], [3.0]]
        actual = [[1.4, 2.8, 1.4], [2.8]]
        metric_result = absolute_error.evaluate(expected, actual)
        value = metric_result.get_value(Score, "score").value
        self.assertAlmostEqual(value, math.sqrt(0.1))

    def test_negative(self) -> None:
        absolute_error = narrow(
            RootMeanSquaredErrorMetaEval,
            RootMeanSquaredErrorMetaEvalConfig(negative=True).to_metric(),
        )
        expected = [[1.0], [3.0, 1.0, 3.0]]
        actual = [[1.4], [2.8, 1.4, 2.8]]
        metric_result = absolute_error.evaluate(expected, actual)
        value = metric_result.get_value(Score, "score").value
        self.assertAlmostEqual(value, -math.sqrt(0.1))


class AbsoluteErrorMetaEvalConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            AbsoluteErrorMetaEvalConfig().serialize(),
            {
                "source_language": None,
                "target_language": None,
                "negative": False,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            AbsoluteErrorMetaEvalConfig.deserialize({}),
            AbsoluteErrorMetaEvalConfig(),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            AbsoluteErrorMetaEvalConfig().to_metric(),
            AbsoluteErrorMetaEval,
        )


class AbsoluteErrorMetaEvalTest(unittest.TestCase):
    def test_basic_calculation(self) -> None:
        absolute_error = narrow(
            AbsoluteErrorMetaEval, AbsoluteErrorMetaEvalConfig().to_metric()
        )
        expected = [[1.0, 2.0], [3.0]]
        actual = [[1.5, 2.0], [3.7]]
        metric_result = absolute_error.evaluate(expected, actual)
        value = metric_result.get_value(Score, "score").value
        self.assertAlmostEqual(value, 0.4)

    def test_negative(self) -> None:
        absolute_error = narrow(
            AbsoluteErrorMetaEval,
            AbsoluteErrorMetaEvalConfig(negative=True).to_metric(),
        )
        expected = [[1.0], [2.0, 3.0]]
        actual = [[1.5], [2.0, 3.7]]
        metric_result = absolute_error.evaluate(expected, actual)
        value = metric_result.get_value(Score, "score").value
        self.assertAlmostEqual(value, -0.4)
