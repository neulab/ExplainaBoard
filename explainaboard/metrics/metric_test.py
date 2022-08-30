"""Tests for explainaboard.metrics.metric"""

from __future__ import annotations

import unittest

from explainaboard.metrics.metric import MetricConfig


class MetricConfigTest(unittest.TestCase):
    def test_serialize(self) -> None:
        self.assertEqual(
            MetricConfig("foo").serialize(),
            {
                "name": "foo",
                "source_language": None,
                "target_language": None,
                "cls_name": "MetricConfig",
                "external_stats": None,
            },
        )

    def test_deserialize(self) -> None:
        self.assertEqual(
            MetricConfig.deserialize({"name": "foo"}),
            MetricConfig("foo"),
        )

    def test_to_metric(self) -> None:
        with self.assertRaises(NotImplementedError):
            MetricConfig("foo").to_metric()
