"""Tests for explainaboard.metrics.external_eval"""

from __future__ import annotations
import unittest

from explainaboard.metrics.external_eval import (
    ExternalEval,
    ExternalEvalConfig,
)


class ExternalEvalConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            ExternalEvalConfig(
                "ExternalEval",
                aspect="foo",
                n_annotators=1,
                categories=2,
                instruction="instruction",
            ).serialize(),
            {
                "name": "ExternalEval",
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
                    "name": "ExternalEval",
                    "aspect": "foo",
                    "n_annotators": 1,
                    "categories": 2,
                    "instruction": "instruction",
                }
            ),
            ExternalEvalConfig(
                "ExternalEval",
                aspect="foo",
                n_annotators=1,
                categories=2,
                instruction="instruction",
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            ExternalEvalConfig("ExternalEval").to_metric(),
            ExternalEval,
        )
