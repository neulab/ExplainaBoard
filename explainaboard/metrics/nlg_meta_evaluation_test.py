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
    def test_serialize(self) -> None:
        self.assertEqual(
            CorrelationConfig(
                "Correlation",
                group_by="system",
                use_z_score=True,
                no_human=True,
            ).serialize(),
            {
                "name": "Correlation",
                "source_language": None,
                "target_language": None,
                "cls_name": "CorrelationConfig",
                "external_stats": None,
                "group_by": "system",
                "use_z_score": True,
                "no_human": True,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            CorrelationConfig.deserialize(
                {
                    "name": "Correlation",
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                }
            ),
            CorrelationConfig(
                "Correlation",
                group_by="system",
                use_z_score=True,
                no_human=True,
            ),
        )


class KtauCorrelationConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            KtauCorrelationConfig(
                "KtauCorrelation",
                group_by="system",
                use_z_score=True,
                no_human=True,
                threshold=25,
            ).serialize(),
            {
                "name": "KtauCorrelation",
                "source_language": None,
                "target_language": None,
                "cls_name": "KtauCorrelationConfig",
                "external_stats": None,
                "group_by": "system",
                "use_z_score": True,
                "no_human": True,
                "threshold": 25,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            KtauCorrelationConfig.deserialize(
                {
                    "name": "KtauCorrelation",
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                    "threshold": 25,
                }
            ),
            KtauCorrelationConfig(
                "KtauCorrelation",
                group_by="system",
                use_z_score=True,
                no_human=True,
                threshold=25,
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            KtauCorrelationConfig("KtauCorrelation").to_metric(),
            KtauCorrelation,
        )


class PearsonCorrelationConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            PearsonCorrelationConfig(
                "PearsonCorrelation",
                group_by="system",
                use_z_score=True,
                no_human=True,
            ).serialize(),
            {
                "name": "PearsonCorrelation",
                "source_language": None,
                "target_language": None,
                "cls_name": "PearsonCorrelationConfig",
                "external_stats": None,
                "group_by": "system",
                "use_z_score": True,
                "no_human": True,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            PearsonCorrelationConfig.deserialize(
                {
                    "name": "PearsonCorrelation",
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                }
            ),
            PearsonCorrelationConfig(
                "PearsonCorrelation",
                group_by="system",
                use_z_score=True,
                no_human=True,
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            PearsonCorrelationConfig("PearsonCorrelation").to_metric(),
            PearsonCorrelation,
        )
