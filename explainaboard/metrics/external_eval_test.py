"""Tests for explainaboard.metrics.external_eval"""

from __future__ import annotations

import unittest

import numpy as np

from explainaboard.metrics.external_eval import (
    ExternalEval,
    ExternalEvalConfig,
    UNANNOTATED_SYMBOL,
)


class ExternalEvalConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            ExternalEvalConfig(
                aspect="foo",
                n_annotators=1,
                categories=2,
                instruction="instruction",
            ).serialize(),
            {
                "source_language": None,
                "target_language": None,
                "external_stats": None,
                "aspect": "foo",
                "n_annotators": 1,
                "categories": 2,
                "instruction": "instruction",
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            ExternalEvalConfig.deserialize(
                {
                    "aspect": "foo",
                    "n_annotators": 1,
                    "categories": 2,
                    "instruction": "instruction",
                }
            ),
            ExternalEvalConfig(
                aspect="foo",
                n_annotators=1,
                categories=2,
                instruction="instruction",
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            ExternalEvalConfig().to_metric(),
            ExternalEval,
        )


class ExternalEvalTest(unittest.TestCase):
    def test_calc_stats_from_external(self) -> None:
        config = ExternalEvalConfig(
            n_annotators=2, external_stats=np.array([[1, 2], [3, 4]])
        )
        metric = ExternalEval(config)
        stats = metric.calc_stats_from_external()
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats.num_statistics(), 2)
        np.testing.assert_array_equal(stats.get_data(), np.array([[1, 2], [3, 4]]))

    def test_calc_stats_from_data_with_external_stats(self) -> None:
        config = ExternalEvalConfig(n_annotators=1, external_stats=np.array([[1], [0]]))
        metric = ExternalEval(config)
        true_data = [1, 0]
        pred_data = [1, 1]
        stats = metric.calc_stats_from_data(true_data, pred_data)
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats.num_statistics(), 1)
        np.testing.assert_array_equal(stats.get_data(), np.array([[1], [0]]))

    def test_calc_stats_from_data_without_external_stats(self) -> None:
        config = ExternalEvalConfig(n_annotators=1)
        metric = ExternalEval(config)
        true_data = [1, 0]
        pred_data = [1, 1]
        stats = metric.calc_stats_from_data(true_data, pred_data)
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats.num_statistics(), 1)
        np.testing.assert_array_equal(
            stats.get_data(), np.array([[UNANNOTATED_SYMBOL], [UNANNOTATED_SYMBOL]])
        )
