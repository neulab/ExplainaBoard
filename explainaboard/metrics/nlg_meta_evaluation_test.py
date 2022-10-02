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
            CorrelationConfig.deserialize(
                {
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                }
            ),
            CorrelationConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
            ),
        )


class KtauCorrelationConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            KtauCorrelationConfig(
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
            KtauCorrelationConfig.deserialize(
                {
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                    "threshold": 25,
                }
            ),
            KtauCorrelationConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
                threshold=25,
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            KtauCorrelationConfig().to_metric(),
            KtauCorrelation,
        )


class PearsonCorrelationConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            PearsonCorrelationConfig(
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
            PearsonCorrelationConfig.deserialize(
                {
                    "group_by": "system",
                    "use_z_score": True,
                    "no_human": True,
                }
            ),
            PearsonCorrelationConfig(
                group_by="system",
                use_z_score=True,
                no_human=True,
            ),
        )

    def test_to_metric(self) -> None:
        self.assertIsInstance(
            PearsonCorrelationConfig().to_metric(),
            PearsonCorrelation,
        )
