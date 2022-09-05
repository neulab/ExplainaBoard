"""Tests for explainaboard.metrics.registry"""

from dataclasses import dataclass
import unittest

from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.registry import (
    get_metric_config_serializer,
    metric_config_registry,
)


@dataclass
@metric_config_registry.register("FooConfig")
class FooConfig(MetricConfig):
    def to_metric(self):
        pass


class RegistryTest(unittest.TestCase):
    def test_serialize(self):
        serializer = get_metric_config_serializer()
        self.assertEqual(
            serializer.serialize(FooConfig('Foo')),
            {
                "cls_name": "FooConfig",
                "name": "Foo",
                "source_language": None,
                "target_language": None,
                "external_stats": None,
            },
        )

    def test_deserialize(self):
        serializer = get_metric_config_serializer()
        self.assertEqual(
            serializer.deserialize({"cls_name": "FooConfig", "name": "Foo"}),
            FooConfig('Foo'),
        )
