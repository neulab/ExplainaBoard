"""Tests for explainaboard.metrics.registry"""

from dataclasses import dataclass
import unittest

from explainaboard.metrics.metric import MetricConfig
from explainaboard.metrics.registry import (
    metric_config_from_dict,
    metric_config_registry,
)


@dataclass
@metric_config_registry.register("FooConfig")
class FooConfig(MetricConfig):
    pass


class RegistryTest(unittest.TestCase):
    def test_metric_config_from_dict(self) -> None:
        self.assertEqual(
            metric_config_from_dict({"cls_name": "FooConfig", "name": "Foo"}),
            FooConfig("Foo"),
        )
