"""Tests for explainaboard.metrics.nlg_meta_evaluation"""

from __future__ import annotations

import unittest

from explainaboard.metrics.meta_evaluation import (
    CorrelationWMTDAConfig,
    KtauCorrelationWMTDA,
    KtauCorrelationWMTDAConfig,
    PearsonCorrelationWMTDA,
    PearsonCorrelationWMTDAConfig,
)


class CorrelationConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            CorrelationWMTDAConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
            ).serialize(),
            {
                "source_language": None,
                "target_language": None,
                "group_by": "system",
                "use_z_score": True,
                "no_human": True,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            CorrelationWMTDAConfig.deserialize(
                {
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                }
            ),
            CorrelationWMTDAConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
            ),
        )


class KtauCorrelationWMTDAConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            KtauCorrelationWMTDAConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
                threshold=25,
            ).serialize(),
            {
                "source_language": None,
                "target_language": None,
                "group_by": "system",
                "use_z_score": True,
                "no_human": True,
                "threshold": 25,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            KtauCorrelationWMTDAConfig.deserialize(
                {
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                    "threshold": 25,
                }
            ),
            KtauCorrelationWMTDAConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
                threshold=25,
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            KtauCorrelationWMTDAConfig().to_metric(),
            KtauCorrelationWMTDA,
        )


class PearsonCorrelationConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            PearsonCorrelationWMTDAConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
            ).serialize(),
            {
                "source_language": None,
                "target_language": None,
                "group_by": "system",
                "use_z_score": True,
                "no_human": True,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            PearsonCorrelationWMTDAConfig.deserialize(
                {
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                }
            ),
            PearsonCorrelationWMTDAConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            PearsonCorrelationWMTDAConfig().to_metric(),
            PearsonCorrelationWMTDA,
        )
