"""Tests for explainaboard.metrics.external_eval"""

from __future__ import annotations

import unittest

import numpy as np

from explainaboard.metrics.external_eval import (
    ExternalEval,
    ExternalEvalConfig,
    UNANNOTATED_SYMBOL,
)
from explainaboard.utils.typing_utils import narrow


class ExternalEvalConfigTest(unittest.TestCase):
    def test_to_metric(self) -> None:
        self.assertIsInstance(
            ExternalEvalConfig("ExternalEval").to_metric(),
            ExternalEval,
        )


class ExternalEvalTest(unittest.TestCase):
    def test_calc_stats_from_external(self) -> None:
        metric = narrow(
            ExternalEval, ExternalEvalConfig("ExternalEval", n_annotators=2).to_metric()
        )

        stats = metric.calc_stats_from_external(
            ExternalEvalConfig(
                "ExternalEval",
                n_annotators=2,
                external_stats=np.array([[1, 2], [3, 4]]),
            )
        )
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats.num_statistics(), 2)
        np.testing.assert_array_equal(stats.get_data(), np.array([[1, 2], [3, 4]]))

    def test_calc_stats_from_data_with_external_stats(self) -> None:
        metric = ExternalEvalConfig("ExternalEval", n_annotators=2).to_metric()
        true_data = [1, 0]
        pred_data = [1, 1]
        stats = metric.calc_stats_from_data(
            true_data,
            pred_data,
            ExternalEvalConfig(
                "ExternalEval", n_annotators=1, external_stats=np.array([[1], [0]])
            ),
        )
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats.num_statistics(), 1)
        np.testing.assert_array_equal(stats.get_data(), np.array([[1], [0]]))

    def test_calc_stats_from_data_without_external_stats(self) -> None:
        metric = ExternalEvalConfig("ExternalEval", n_annotators=2).to_metric()
        true_data = [1, 0]
        pred_data = [1, 1]
        stats = metric.calc_stats_from_data(
            true_data,
            pred_data,
            ExternalEvalConfig("ExternalEval", n_annotators=1),
        )
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats.num_statistics(), 1)
        np.testing.assert_array_equal(
            stats.get_data(), np.array([[UNANNOTATED_SYMBOL], [UNANNOTATED_SYMBOL]])
        )
